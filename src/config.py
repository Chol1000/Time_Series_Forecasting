from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR   = PROJECT_ROOT / "outputs" / "figures"
MODELS_DIR    = PROJECT_ROOT / "outputs" / "models"
METRICS_DIR   = PROJECT_ROOT / "outputs" / "metrics"
TABLES_DIR    = PROJECT_ROOT / "outputs" / "tables"

FIG_T1 = FIGURES_DIR / "task1"
FIG_T2 = FIGURES_DIR / "task2"
FIG_T3 = FIGURES_DIR / "task3"

PARQUET_PATH     = PROCESSED_DIR / "traffic_wide.parquet"
TARGET_AREAS_CSV = PROCESSED_DIR / "target_areas.csv"

GLOBAL_SEED  = 42
SEQ_LEN      = 144
