# Milan Mobile Network Traffic — Time Series Forecasting

**Dataset:** [TIM Milan · Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV) · Nov 2013 – Jan 2014 · 10,000 grid squares · 10-min intervals

---

Three forecasting models — Holt-Winters, LSTM, and an encoder-only Transformer — are trained and compared on real mobile internet traffic from Telecom Italia's Milan network. The test week is 16–22 December 2013.

The short version of the result: a classical statistical model from 1957 stays competitive with both neural networks and wins outright on the busiest area in the city.

---

## Results

Rolling one-step-ahead evaluation · 1,008 test steps · all values on held-out test data only.

### Square 5161 — City centre (100th percentile, highest traffic)

| Model | MAE | MAPE | RMSE | Train time |
|-------|-----|------|------|------------|
| Seasonal Naïve *(baseline)* | 338.6 | 25.94% | 619.0 | — |
| **Holt-Winters** *(best)* | **81.88** | **8.40%** | **121.65** | 1.4 s |
| LSTM | 84.19 | 8.72% | 127.89 | 10.5 s |
| Transformer | 100.53 | 14.88% | 133.96 | 61.1 s |

### Square 4159 — Mixed residential (95.8th percentile)

| Model | MAE | MAPE | RMSE | Train time |
|-------|-----|------|------|------------|
| Seasonal Naïve *(baseline)* | 51.2 | 21.80% | 84.6 | — |
| Holt-Winters | 16.91 | 7.39% | 23.14 | 1.4 s |
| **LSTM** *(best)* | **15.09** | **6.70%** | **20.37** | 10.5 s |
| Transformer | 17.95 | 8.87% | 22.42 | 61.1 s |

### Square 4556 — Sub-central commercial (98.9th percentile)

| Model | MAE | MAPE | RMSE | Train time |
|-------|-----|------|------|------------|
| Seasonal Naïve *(baseline)* | 76.3 | 17.46% | 108.4 | — |
| Holt-Winters | 27.74 | 6.42% | 37.12 | 1.4 s |
| LSTM | 28.89 | 6.67% | 38.38 | 10.5 s |
| **Transformer** *(best)* | **26.72** | **6.17%** | **36.03** | 61.1 s |

All three models beat the seasonal naïve baseline by **65–80% in RMSE**. Diebold-Mariano tests confirm these rankings are statistically significant (p < 0.001 on Areas 4159 and 4556; p = 0.005 on Square 5161).

**Why Holt-Winters holds up on the busiest area:** Square 5161's traffic is driven by a strong, stable 24-hour seasonal pattern — office hours, evening peaks, near-silence at 03:00. A fixed seasonal template with a damped trend fits this structure almost perfectly. The neural models only pull ahead when there is enough variation in the lower-traffic areas for their extra capacity to help.

---

## What the Three Notebooks Cover

| Notebook | Task |
|----------|------|
| `task1_data_handling.ipynb` | Loads 62 daily text files (~5 GB raw). Cuts peak memory by 80% through float32 casting, per-file aggregation, and selective column loading. Builds a Parquet cache for downstream notebooks. |
| `task2_eda.ipynb` | Explores seasonal structure (daily period=144, weekly period=1,008), spatial traffic distribution, stationarity (ADF + KPSS), STL decomposition, ACF/PACF, and anomaly detection. |
| `task3_forecasting.ipynb` | Runs 4 tuning experiments per model, selects the best config on the Dec 9–15 validation window, retrains on the full train+val set, and evaluates on Dec 16–22. |

---

## Setup

### What you need

- Python 3.9 or later
- Git LFS (for the Parquet cache — see Step 1)
- **~2 GB RAM** recommended (loading the Parquet pushes peak RSS to ~1.5 GB; 8 GB total is comfortable)
- GPU optional — Apple Silicon MPS, NVIDIA CUDA, or CPU all auto-detected

### Step 1 — Install Git LFS, then clone

The processed Parquet cache is stored via Git LFS. Install LFS first so it downloads with the repo:

```bash
# macOS
brew install git-lfs

# Linux (Debian/Ubuntu)
sudo apt install git-lfs

# Windows — download installer from https://git-lfs.com
git lfs install
```

Then clone:

```bash
git clone https://github.com/Chol1000/Time_Series_Forecasting.git
cd Time_Series_Forecasting
```

### Step 2 — Virtual environment

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Data

**Minimum data to run the prediction (Task 3 only):**
Nothing else needed. All saved model weights and the processed Parquet cache are already in the repository. Clone and run Task 3 directly.

**To reproduce everything from Task 1 (optional):**
Download the 62 daily `.txt` files from Harvard Dataverse and place them in `data/raw/`:

- Traffic data: [doi:10.7910/DVN/EGZHFV](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV)
- Grid GeoJSON *(optional — only needed for the geographic spatial map in Task 2)*: [doi:10.7910/DVN/QJWLFU](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QJWLFU)

Files must follow the naming pattern `sms-call-internet-mi-YYYY-MM-DD.txt`. The raw files are excluded from git due to size.

---

## Running

> Run in order. Task 1 must go first — it builds the Parquet cache that Tasks 2 and 3 read. Tasks 2 and 3 can be re-run independently once the cache exists.

### Option A — Jupyter (interactive)

```bash
jupyter notebook
```

Open each notebook and run via **Kernel → Restart & Run All**:

1. `notebooks/task1_data_handling.ipynb`
2. `notebooks/task2_eda.ipynb`
3. `notebooks/task3_forecasting.ipynb`

### Option B — Google Colab (no local install needed)

Open any notebook directly from GitHub:

| Notebook | Link |
|----------|------|
| Task 1 — Data Handling | [Open in Colab ↗](https://colab.research.google.com/github/Chol1000/Time_Series_Forecasting/blob/main/notebooks/task1_data_handling.ipynb) |
| Task 2 — EDA | [Open in Colab ↗](https://colab.research.google.com/github/Chol1000/Time_Series_Forecasting/blob/main/notebooks/task2_eda.ipynb) |
| Task 3 — Forecasting | [Open in Colab ↗](https://colab.research.google.com/github/Chol1000/Time_Series_Forecasting/blob/main/notebooks/task3_forecasting.ipynb) |

**Before running any cells**, insert and run this as the very first cell:

```python
# Clone the repo — LFS downloads the Parquet cache automatically (~468 MB, ~1–2 min)
!git lfs install
!git clone https://github.com/Chol1000/Time_Series_Forecasting.git
import os
os.chdir('/content/Time_Series_Forecasting')
!pip install -r requirements.txt
```

Then enable GPU for Task 3: **Runtime → Change runtime type → T4 GPU → Save**

After that, run the remaining cells normally. Task 3 takes ~20–30 min on a T4 GPU.

> **Colab notes:**
> - To rerun Task 1 from raw data (not cache): upload the `.txt` files to `/content/Time_Series_Forecasting/data/raw/` via the Files panel on the left.
> - Colab sessions disconnect after ~90 min of inactivity — keep the tab active during long Task 3 runs.
> - If the clone fails on LFS, run `!apt-get install -y git-lfs` first, then retry.

---

### Option C — Command line (no browser)

**macOS / Linux:**
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

**Windows (PowerShell):**
```powershell
jupyter nbconvert --to notebook --execute notebooks/task1_data_handling.ipynb `
  --output task1_data_handling.ipynb --output-dir notebooks/ `
  --ExecutePreprocessor.timeout=600

jupyter nbconvert --to notebook --execute notebooks/task2_eda.ipynb `
  --output task2_eda.ipynb --output-dir notebooks/ `
  --ExecutePreprocessor.timeout=600

jupyter nbconvert --to notebook --execute notebooks/task3_forecasting.ipynb `
  --output task3_forecasting.ipynb --output-dir notebooks/ `
  --ExecutePreprocessor.timeout=7200
```

### Expected runtimes

| Step | Apple Silicon / CUDA | CPU only |
|------|---------------------|----------|
| Task 1 — first run (builds cache) | ~2 min | ~2 min |
| Task 1 — subsequent runs (from cache) | ~3 s | ~3 s |
| Task 2 — full EDA | ~1 min | ~1 min |
| Task 3 — full training (4 experiments × 3 models) | ~15–20 min | ~45–60 min |

Task 3 detects existing `.pt` weight files and skips retraining automatically — re-running it just regenerates plots.

---

## Repository Structure

```
Time_Series_Forecasting/
│
├── notebooks/
│   ├── task1_data_handling.ipynb     # Data loading and memory optimisation
│   ├── task2_eda.ipynb               # Exploratory data analysis
│   └── task3_forecasting.ipynb       # Model training and evaluation
│
├── src/
│   ├── config.py                     # Central paths and constants
│   ├── data_loader.py                # Dataset ingestion pipeline
│   ├── models.py                     # Holt-Winters, LSTM, Transformer + eval helpers
│   └── __init__.py
│
├── outputs/
│   ├── figures/
│   │   ├── task1/                    # Memory and zero-value analysis
│   │   ├── task2/                    # EDA: decomposition, ACF/PACF, heatmaps, anomalies
│   │   └── task3/                    # Predictions, learning curves, failure analysis
│   ├── models/
│   │   ├── lstm/                     # lstm_4159.pt, lstm_4556.pt, lstm_5161.pt
│   │   ├── transformer/              # transformer_XXXX.pt (same three areas)
│   │   └── holtwinters/              # hw_XXXX.json (fitted parameters per area)
│   ├── metrics/
│   │   └── all_model_metrics.csv     # MAE, RMSE, MAPE, sMAPE — all models × areas
│   └── tables/
│       └── timing_table.csv          # Training and inference times
│
├── data/
│   ├── raw/                          # Not tracked — download from Harvard Dataverse
│   └── processed/                    # Parquet cache tracked via Git LFS; CSV tracked directly
│                                     # (auto-built by Task 1 if not present)
│
├── requirements.txt
├── .gitignore
└── README.md
```

All outputs (figures, model weights, metrics) are committed and viewable without running any code.

---

## Sample Outputs

### Task 1 — Memory optimisation

![Memory comparison](outputs/figures/task1/task1_memory_comparison.png)

Float32 casting cuts the in-memory footprint in half versus pandas' float64 default. Combined with groupby aggregation (70% row reduction) and loading only 3 of 8 columns, the full 62-file matrix fits in 357 MB instead of ~714 MB.

---

### Task 2 — Traffic patterns

![Two-week time series](outputs/figures/task2/task2_timeseries_2weeks.png)

The 24-hour seasonal cycle (period = 144 intervals) is visible in all three areas. Square 5161 adds a secondary business-hours shoulder that the other two areas — both more residential — do not show.

![Seasonal decomposition](outputs/figures/task2/task2_decomposition_5161.png)

STL decomposition for Square 5161 cleanly separates the daily seasonal component from the residuals. The trend shows a mild rise through November–December before a sharp Christmas dip as the office population temporarily leaves the city centre.

![Spatial heatmap](outputs/figures/task2/task2_spatial_heatmap.png)

Traffic is heavily concentrated in central Milan. The Duomo–Stazione Centrale corridor accounts for a disproportionate share of city-wide CDR activity — Square 5161 sits at the very top of this cluster.

---

### Task 3 — Forecasting

![Train/test split](outputs/figures/task3/task3_train_test_split.png)

Data split: tune-train (1 Nov – 8 Dec, 5,472 steps), validation (9–15 Dec, 1,008 steps), full-train (1 Nov – 15 Dec, 6,480 steps), test (16–22 Dec, 1,008 steps). No test data was seen during training or hyperparameter selection.

![LSTM predictions](outputs/figures/task3/task3_predictions_lstm.png)

Rolling one-step-ahead predictions from the LSTM on the test week. The model tracks daily peaks well but under-predicts the Saturday 21 December surge — a recurring failure mode shared by all three models when traffic deviates from the training distribution.

![Failure analysis](outputs/figures/task3/task3_failure_analysis.png)

The worst prediction windows fall on Tuesday 17 December (an unusual early-week spike) and Sunday 22 December evening (the pre-Christmas surge). Both events sit outside the range the models were trained on. This is an inherent limitation of univariate forecasting without calendar or event signals.

---

## Reproducibility Notes

- **Pre-run outputs are committed.** All figures, saved model weights, and metrics are in the repository. No code needs to run to review results.
- **Parquet cache committed via LFS.** Task 1 can be skipped if the LFS file downloads correctly on clone.
- **Fixed random seeds.** PyTorch seeds are set at the start of each training cell. Minor variation (±2%) may occur from hardware-level non-determinism on MPS/CUDA.
- **Auto-regressive evaluation.** Results from feeding predictions back as input (auto-regressive rollout) inflate RMSE by 3.4–13.5× and are reported as a diagnostic comparison only — not used in the main evaluation.

---

## References

1. G. Barlacchi et al., "A multi-source dataset of urban life in the city of Milan and the Province of Trentino," *Scientific Data*, vol. 2, p. 150055, 2015. [doi:10.1038/sdata.2015.55](https://doi.org/10.1038/sdata.2015.55)
2. S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.
3. A. Vaswani et al., "Attention Is All You Need," in *Advances in Neural Information Processing Systems*, 2017.
4. C. Holt, "Forecasting seasonals and trends by exponentially weighted moving averages," *ONR Research Memorandum*, 1957.
5. F. X. Diebold and R. S. Mariano, "Comparing Predictive Accuracy," *Journal of Business & Economic Statistics*, vol. 13, no. 3, pp. 253–263, 1995.
6. GitHub repository: [github.com/Chol1000/Time_Series_Forecasting](https://github.com/Chol1000/Time_Series_Forecasting)
