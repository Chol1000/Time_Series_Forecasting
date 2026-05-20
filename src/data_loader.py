"""
data_loader.py
--------------
Efficient ingestion and memory-optimised processing of the Milan TIM dataset.

Strategy overview
-----------------
1. Column selection at read time  — only square_id (col 0), timestamp (col 1),
   and internet CDR (col 7) are loaded; the 5 SMS/call columns are skipped entirely.
2. dtype down-casting             — square_id → int16 (range 1–10 000),
                                    internet   → float32 (halves float64 footprint).
3. Per-file aggregation           — internet traffic is summed across all country
                                    codes for each (square_id, timestamp) pair,
                                    reducing ~4.8 M rows/file to ~1.44 M.
4. Pivot to wide format           — final DataFrame: rows = timestamps, columns = areas.
5. Parquet caching                — processed file saved to data/processed/ so the
                                    62-file pipeline only runs once.
"""

import os
import glob
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

# ── Canonical project root (two levels up from this file: src/ → project root)
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CACHE_PATH = PROCESSED_DIR / "traffic_wide.parquet"

# Only the three columns relevant to this assignment
_USECOLS  = [0, 1, 7]
_COLNAMES = ["square_id", "time_ms", "internet"]
_DTYPES   = {"square_id": "int16", "time_ms": "int64", "internet": "float32"}


# ── Utilities ──────────────────────────────────────────────────────────────────

def get_process_memory_mb() -> float:
    """Return current process RSS in megabytes."""
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def _load_single_file(filepath: str) -> pd.DataFrame:
    """
    Read one daily raw file and aggregate internet CDR per (square_id, timestamp).

    Each raw file contains multiple rows per (square_id, timestamp) — one row per
    country code that generated CDRs in that interval.  We sum across all country
    codes to obtain total internet traffic for each (area, interval) pair.
    """
    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        usecols=_USECOLS,
        names=_COLNAMES,
        dtype=_DTYPES,
    )
    agg = df.groupby(["square_id", "time_ms"], as_index=False)["internet"].sum()
    agg["internet"] = agg["internet"].astype("float32")
    return agg


# ── Main loader ────────────────────────────────────────────────────────────────

def load_and_process_all(force_rebuild: bool = False) -> pd.DataFrame:
    """
    Load, aggregate, and pivot the full Milan dataset into a wide DataFrame.

    Returns
    -------
    pd.DataFrame
        Shape (n_timestamps, 10 000).  Index = timezone-aware timestamp (CET);
        columns = integer square IDs 1–10 000.  dtype = float32.
        Missing (area, interval) pairs are filled with 0.0.

    Parameters
    ----------
    force_rebuild : bool
        If True, ignore any existing Parquet cache and re-process all raw files.
    """
    if CACHE_PATH.exists() and not force_rebuild:
        print(f"[data_loader] Loading cached data from {CACHE_PATH.name} ...")
        wide = pd.read_parquet(CACHE_PATH)
        print(f"[data_loader] Loaded  shape={wide.shape}  "
              f"memory={wide.memory_usage(deep=True).sum()/1e6:.1f} MB")
        return wide

    files = sorted(glob.glob(str(RAW_DIR / "sms-call-internet-mi-*.txt")))
    if not files:
        raise FileNotFoundError(
            f"No raw data files found in {RAW_DIR}. "
            "Please download the TIM dataset and place the .txt files there."
        )
    print(f"[data_loader] Found {len(files)} daily files — starting ingestion ...")

    mem_before = get_process_memory_mb()
    tracemalloc.start()
    t_start = time.perf_counter()

    chunks = []
    for i, fp in enumerate(files):
        t0 = time.perf_counter()
        chunks.append(_load_single_file(fp))
        if (i + 1) % 10 == 0 or i == 0 or i == len(files) - 1:
            print(f"  [{i+1:3d}/{len(files)}] {Path(fp).name:45s} "
                  f"{time.perf_counter()-t0:.1f}s  RSS={get_process_memory_mb():.0f}MB")

    print("[data_loader] Concatenating ...")
    long_df = pd.concat(chunks, ignore_index=True)
    del chunks

    # Final group-by handles any edge timestamps spanning midnight (rare but safe)
    long_df = long_df.groupby(["time_ms", "square_id"], as_index=False)["internet"].sum()
    long_df["internet"] = long_df["internet"].astype("float32")

    # The raw epoch values are UTC milliseconds.  Adding +1 h shifts them to
    # Milan local time (CET = UTC+1, no DST change in the Nov–Jan window).
    # The resulting timestamps are stored with the UTC timezone label so that
    # the parquet index is timezone-aware without carrying a CET/Europe label.
    # Convention throughout this project: index values represent CET wall-clock
    # time; the UTC label is a storage artefact.  Never call tz_convert() on
    # this index — doing so would add a second +1 h offset and misalign the data.
    # All pd.Timestamp cut-points in downstream notebooks must use tz='UTC'
    # to match this convention (e.g. pd.Timestamp('2013-12-09', tz='UTC')).
    long_df["timestamp"] = (
        pd.to_datetime(long_df["time_ms"], unit="ms", utc=True)
        + pd.Timedelta(hours=1)
    )
    long_df.drop(columns="time_ms", inplace=True)

    print("[data_loader] Pivoting to wide format ...")
    wide = long_df.pivot(index="timestamp", columns="square_id", values="internet")
    wide.columns.name = None
    wide = wide.astype("float32")
    wide.fillna(0.0, inplace=True)

    t_total = time.perf_counter() - t_start
    mem_after = get_process_memory_mb()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\n[data_loader] === Memory & Timing Report ===")
    print(f"  Process RSS before  : {mem_before:>8.0f} MB")
    print(f"  Process RSS after   : {mem_after:>8.0f} MB  (+{mem_after-mem_before:.0f} MB)")
    print(f"  Peak traced alloc   : {peak/1e6:>8.0f} MB")
    print(f"  Wide DataFrame      : {wide.memory_usage(deep=True).sum()/1e6:>8.1f} MB  "
          f"(float32, {wide.shape[0]:,} × {wide.shape[1]:,})")
    print(f"  Total elapsed       : {t_total:>8.1f} s\n")

    print(f"[data_loader] Saving Parquet cache → {CACHE_PATH} ...")
    wide.to_parquet(CACHE_PATH)
    return wide


# ── Memory comparison utility ──────────────────────────────────────────────────

def memory_optimization_report(wide: pd.DataFrame) -> dict:
    """
    Compute theoretical memory for several dtype strategies on the same shape.
    Used in the report to evidence the impact of float32 down-casting.

    Returns a dict  {strategy_label: size_mb}.
    """
    n = wide.size          # total elements
    strategies = {
        "float64 (default — baseline)": n * 8 / 1e6,
        "float32 (adopted)":            n * 4 / 1e6,
        "float16 (half precision)":     n * 2 / 1e6,
        "int16  (×100 fixed-point)":    n * 2 / 1e6,
    }
    baseline = strategies["float64 (default — baseline)"]
    print("\n[data_loader] === dtype Memory Comparison ===")
    print(f"  {'Strategy':<35} {'Size (MB)':>10}   {'vs float64':>10}")
    print("  " + "-" * 60)
    for k, v in strategies.items():
        tag = "(adopted)" if "adopted" in k else ""
        print(f"  {k:<35} {v:>10.1f} MB   {(1-v/baseline)*100:>8.1f}% smaller  {tag}")
    return strategies
