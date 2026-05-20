# Time Series Forecasting of Milan Mobile Network Traffic

**Course:** ML Techniques I — Formative Assignment  
**Student:** Chol Monykuch · c.monykuch@alustudent.com  
**Dataset:** TIM Milan Mobile Network Traffic (Nov 2013 – Jan 2014)

---

## Overview

This project analyses and forecasts internet traffic across the Milan mobile network grid using three model families:

| Model | Type | Strategy |
|---|---|---|
| **Holt-Winters** | Statistical | Triple exponential smoothing with additive seasonality |
| **LSTM** | Deep learning | Sequence-to-one recurrent network |
| **Transformer** | Deep learning | Encoder-only Pre-LN architecture |

Evaluation is performed via **rolling one-step-ahead prediction** over a held-out test week (16–22 Dec 2013) on three geographically distinct grid squares.

---

## Repository Structure

```
Time_Series_Forecasting/
├── data/
│   ├── raw/                              ← 62 raw daily .txt files (not tracked — see below)
│   └── processed/                        ← Auto-generated cache (not tracked — see below)
│       ├── traffic_wide.parquet          ←   Wide-format dataset (~468 MB, ~2 min to build)
│       └── target_areas.csv             ←   Three target area IDs (written by Task 1)
├── notebooks/
│   ├── task1_data_handling.ipynb        ← Task 1: Data loading & memory optimisation
│   ├── task2_eda.ipynb                  ← Task 2: Exploratory data analysis
│   └── task3_forecasting.ipynb          ← Task 3: Model design, training & evaluation
├── outputs/
│   ├── figures/
│   │   ├── task1/                       ← Memory comparison chart, zero-value analysis
│   │   ├── task2/                       ← Time-series, decomposition, ACF/PACF, heatmaps
│   │   └── task3/                       ← Prediction plots, training curves, comparison charts
│   ├── models/
│   │   ├── lstm/                        ← Saved LSTM weights (.pt) per area
│   │   ├── transformer/                 ← Saved Transformer weights (.pt) per area
│   │   └── holtwinters/                 ← Saved HW parameters (.json) per area
│   ├── metrics/
│   │   └── all_model_metrics.csv        ← MAE / RMSE / MAPE / sMAPE for all models × areas
│   └── tables/
│       └── timing_table.csv             ← Training and inference time per model
├── src/
│   ├── __init__.py                      ← Public API exports
│   ├── config.py                        ← Centralised paths and constants
│   ├── data_loader.py                   ← Dataset ingestion and memory optimisation
│   └── models.py                        ← HoltWinters, LSTM, Transformer implementations
├── requirements.txt
└── README.md
```

> **Not tracked by git:**  
> `data/raw/` — large raw files, download separately (see Step 2 below)  
> `data/processed/` — auto-generated on first run, reproduced in ~2 minutes  
> `outputs/report_milan_traffic.pdf` — submitted separately

---

## Prerequisites

- **Python 3.9 or later**  
  macOS: `brew install python3` · Linux: `sudo apt install python3 python3-pip` · Windows: [python.org](https://python.org)
- **GPU (optional but recommended):** Apple Silicon MPS, CUDA, or CPU fallback (all auto-detected by PyTorch)

---

## Quickstart

### Step 1 — Clone the repository

```bash
git clone https://github.com/Chol1000/Time_Series_Forecasting.git
cd Time_Series_Forecasting
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Download the raw data

Download all **62 daily `.txt` files** from the Harvard Dataverse TIM dataset and place them in `data/raw/`:

- **Traffic data:** [doi:10.7910/DVN/EGZHFV](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV)  
- **Grid GeoJSON** (optional, for spatial maps): [doi:10.7910/DVN/QJWLFU](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QJWLFU)

The `data/raw/` folder should contain files named `sms-call-internet-mi-YYYY-MM-DD.txt`.

### Step 4 — Run the notebooks in order

The notebooks share data through `data/processed/`. **Task 1 must run first** — it builds the Parquet cache and writes `target_areas.csv` that Tasks 2 and 3 depend on.

#### Option A — Jupyter (interactive)

```bash
jupyter notebook notebooks/task1_data_handling.ipynb
# Run all cells, then open Task 2:
jupyter notebook notebooks/task2_eda.ipynb
# Run all cells, then open Task 3:
jupyter notebook notebooks/task3_forecasting.ipynb
```

#### Option B — Command line (non-interactive, all platforms)

```bash
jupyter nbconvert --to notebook --execute notebooks/task1_data_handling.ipynb \
        --output task1_data_handling.ipynb --output-dir notebooks/ \
        --ExecutePreprocessor.timeout=600

jupyter nbconvert --to notebook --execute notebooks/task2_eda.ipynb \
        --output task2_eda.ipynb --output-dir notebooks/ \
        --ExecutePreprocessor.timeout=600

jupyter nbconvert --to notebook --execute notebooks/task3_forecasting.ipynb \
        --output task3_forecasting.ipynb --output-dir notebooks/ \
        --ExecutePreprocessor.timeout=7200
```

> **Timing expectations**  
> Task 1 (first run): ~2 minutes to process all 62 files and build the Parquet cache.  
> Task 1 (subsequent runs): ~3 seconds (loads from cache).  
> Task 3 training: ~15–20 minutes on Apple Silicon MPS / CUDA · ~45–60 minutes on CPU.

---

## Expected Outputs After Full Execution

| Location | Contents |
|---|---|
| `outputs/figures/task1/` | Memory comparison bar chart, zero-value distribution |
| `outputs/figures/task2/` | PDF, time-series, rolling stats, decomposition, ACF/PACF, spatial heatmaps, diurnal patterns, anomalies |
| `outputs/figures/task3/` | Prediction plots (9), training curves (6), model comparison, failure analysis, bias/error diagnostics |
| `outputs/models/lstm/` | `lstm_4159.pt`, `lstm_4556.pt`, `lstm_5161.pt` |
| `outputs/models/transformer/` | `transformer_4159.pt`, `transformer_4556.pt`, `transformer_5161.pt` |
| `outputs/models/holtwinters/` | `hw_4159.json`, `hw_4556.json`, `hw_5161.json` |
| `outputs/metrics/all_model_metrics.csv` | MAE, RMSE, MAPE, sMAPE for all 9 model-area combinations |
| `outputs/tables/timing_table.csv` | Training and inference time per model |

---

## Key Results (Test Week: 16–22 Dec 2013)

| Square | Best Model | RMSE | MAPE |
|---|---|---|---|
| 5161 (busiest) | Holt-Winters | ~156 | ~8.25% |
| 4159 | LSTM | ~79 | ~6.5% |
| 4556 | Transformer | ~36 | ~6.0% |

All three models outperform the rolling-mean baseline by **65–80% in RMSE** on the test set. LSTM and Transformer training used early stopping with a 7-day validation window (Dec 9–15).

---

## Reproducibility Notes

- **Random seeds:** PyTorch seeds are fixed in each notebook cell. MPS/CUDA hardware non-determinism may cause ±2% metric variation across runs.
- **Parquet cache:** Once built by Task 1, the file can be retained to skip reprocessing on subsequent runs.
- **Model weights:** Saved checkpoints (`outputs/models/`) allow Task 3 evaluation figures to be reproduced without retraining (the notebook detects existing weights automatically).
- **Timestamp convention:** The Parquet index stores CET wall-clock times under the UTC timezone label (a known storage artefact). Do not call `tz_convert()` on this index — all cut-point timestamps in the notebooks use `tz='UTC'` to match this convention.

---

## References

1. G. Barlacchi *et al.*, "A multi-source dataset of urban life in the city of Milan," *Scientific Data*, 2015.
2. S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, 1997.
3. A. Vaswani *et al.*, "Attention Is All You Need," *NeurIPS*, 2017.
4. C. Holt, "Forecasting seasonals and trends by exponentially weighted moving averages," *ONR Research Memorandum*, 1957.
