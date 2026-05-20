"""
models.py
---------
Three forecasting models for one-step-ahead mobile network traffic prediction.

Models
------
1. HoltWintersModel   — Classical triple exponential smoothing (Holt-Winters)
2. LSTMForecaster     — Stacked LSTM (PyTorch), rolling one-step-ahead inference
3. TransformerForecaster — Encoder-only Transformer (PyTorch), rolling one-step-ahead inference

All NN models:
  • Normalize input per area (zero-mean, unit-variance on training set)
  • Use Adam + ReduceLROnPlateau + early stopping
  • Save best checkpoint by validation MSE
  • Auto-select MPS (Apple Silicon) → CUDA → CPU acceleration
"""

import os
import json
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from statsmodels.tsa.holtwinters import ExponentialSmoothing

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
MODELS_DIR    = PROJECT_ROOT / "outputs" / "models"
METRICS_DIR   = PROJECT_ROOT / "outputs" / "metrics"
TABLES_DIR    = PROJECT_ROOT / "outputs" / "tables"

for _d in [MODELS_DIR / "lstm", MODELS_DIR / "transformer", MODELS_DIR / "holtwinters",
           METRICS_DIR, TABLES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ── Evaluation metrics ─────────────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%).  Zero-valued actuals are excluded."""
    mask = np.abs(y_true) > eps
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / (np.abs(y_true[mask]) + eps))) * 100)

def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Symmetric MAPE (%).  Bounded [0, 200%]; robust when actuals are near zero."""
    return float(np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE":   mae(y_true, y_pred),
        "RMSE":  rmse(y_true, y_pred),
        "MAPE":  mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }


# ── Select best available accelerator ─────────────────────────────────────────

def _best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Holt-Winters Exponential Smoothing
# ══════════════════════════════════════════════════════════════════════════════

class HoltWintersModel:
    """
    Triple Exponential Smoothing with additive trend and additive seasonality.

    The model decomposes the time series into three components:
        Level   ℓ_t = α (y_t − s_{t−m}) + (1−α)(ℓ_{t−1} + φ·b_{t−1})
        Trend   b_t = β (ℓ_t − ℓ_{t−1}) + (1−β) φ·b_{t−1}
        Season  s_t = γ (y_t − ℓ_t)     + (1−γ) s_{t−m}
        Forecast ŷ_{t+h} = ℓ_t + (φ+φ²+…+φ^h)·b_t + s_{t−m+h}
    When damped_trend=False, φ=1 and the equations reduce to standard Holt-Winters.

    Parameters α, β, γ ∈ [0,1] and φ ∈ (0,1] are estimated by minimising SSE on the
    training set using the L-BFGS-B optimiser.

    Parameters
    ----------
    seasonal_periods : int
        Number of time steps per seasonal cycle.  144 = one day at 10-min resolution.
    seasonal : str
        Seasonal component type — 'add' (additive, default) or 'mul' (multiplicative).
    damped_trend : bool
        If True, apply φ-damping to the trend component (Gardner & McKenzie, 1985).
    """

    def __init__(self, seasonal_periods: int = 144, seasonal: str = "add",
                 damped_trend: bool = False):
        self.seasonal_periods = seasonal_periods
        self.seasonal         = seasonal
        self.damped_trend     = damped_trend
        self.result_          = None
        self.train_time_      = 0.0
        self.forecast_time_   = 0.0

    def fit(self, train_series: np.ndarray) -> "HoltWintersModel":
        t0 = time.perf_counter()
        model = ExponentialSmoothing(
            train_series,
            trend="add",
            damped_trend=self.damped_trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            initialization_method="estimated",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.result_ = model.fit(optimized=True, remove_bias=True)
        self.train_time_ = time.perf_counter() - t0
        return self

    def predict(self, n_steps: int) -> np.ndarray:
        """Multi-step ahead forecast from end of training (for failure analysis)."""
        t0 = time.perf_counter()
        fc = self.result_.forecast(n_steps)
        self.forecast_time_ = time.perf_counter() - t0
        return np.asarray(fc)

    def predict_rolling(self, train_series: np.ndarray, test_series: np.ndarray) -> np.ndarray:
        """
        Rolling one-step-ahead forecast (standard evaluation protocol).
        Parameters (alpha, beta, gamma) are fixed from training.
        State (level, trend, seasonal) is updated after each actual observation
        using the Holt-Winters recurrence equations, preventing error accumulation.
        Fast: O(n_test) — no re-fitting.
        """
        t0 = time.perf_counter()
        r    = self.result_
        alp  = float(r.params["smoothing_level"])
        bet  = float(r.params["smoothing_trend"])
        gam  = float(r.params["smoothing_seasonal"])
        m    = self.seasonal_periods
        # initialise state from end of training (numpy arrays in statsmodels)
        lvl  = float(r.level[-1])
        trnd = float(r.trend[-1])
        # last m seasonal indices (ring buffer indexed by position mod m)
        seas = list(r.season[-m:].astype(float))   # length m

        # statsmodels stores non-optimised params as nan; damping_trend is nan
        # for non-damped models (phi conceptually = 1.0 in that case).
        _phi = float(self.result_.params.get("damping_trend", 1.0))
        phi  = 1.0 if np.isnan(_phi) else _phi
        preds = []
        for t, y in enumerate(test_series.astype(np.float64)):
            s_prev = seas[t % m]               # seasonal index m steps back
            # one-step-ahead forecast before observing y_t
            forecast = lvl + phi * trnd + s_prev
            preds.append(max(0.0, forecast))
            # update state with actual observation y_t
            lvl_new  = alp * (y - s_prev) + (1 - alp) * (lvl + phi * trnd)
            trnd_new = bet * (lvl_new - lvl) + (1 - bet) * phi * trnd
            seas[t % m] = gam * (y - lvl_new) + (1 - gam) * s_prev
            lvl, trnd = lvl_new, trnd_new

        self.forecast_time_ = time.perf_counter() - t0
        return np.array(preds, dtype=np.float64)

    def get_params(self) -> dict:
        if self.result_ is None:
            return {}
        p = {
            "alpha": round(float(self.result_.params["smoothing_level"]), 4),
            "beta":  round(float(self.result_.params["smoothing_trend"]), 4),
            "gamma": round(float(self.result_.params["smoothing_seasonal"]), 4),
            "seasonal_periods": self.seasonal_periods,
        }
        if self.damped_trend:
            p["phi"] = round(float(self.result_.params.get("damping_trend", 1.0)), 4)
        return p

    def save(self, path: Path):
        params = self.get_params()
        params["train_time_s"] = round(self.train_time_, 4)
        with open(path, "w") as f:
            json.dump(params, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  LSTM
# ══════════════════════════════════════════════════════════════════════════════

class _SlidingWindowDataset(Dataset):
    """Convert a 1-D time series into (input_window, next_value) sample pairs."""

    def __init__(self, series: np.ndarray, seq_len: int):
        x, y = [], []
        for i in range(len(series) - seq_len):
            x.append(series[i: i + seq_len])
            y.append(series[i + seq_len])
        self.X = torch.from_numpy(np.array(x, dtype=np.float32))
        self.y = torch.from_numpy(np.array(y, dtype=np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _run_epoch(model, loader, criterion, device, optimizer=None):
    """Single training or evaluation epoch.  Returns mean loss."""
    training = optimizer is not None
    model.train() if training else model.eval()
    total = 0.0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item() * len(xb)
    return total / len(loader.dataset)


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class _LSTMNet(nn.Module):
    """
    Stacked LSTM followed by a linear regression head.

    Input  : (batch, seq_len)        — normalised scalar traffic values
    Output : (batch, 1)              — next-step prediction
    """

    def __init__(self, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x.unsqueeze(-1))   # (B, T, H)
        return self.head(out[:, -1, :])        # (B, 1)


class LSTMForecaster:
    """
    Training wrapper for _LSTMNet.

    Normalization is fitted on the training series (mean, std) and applied
    consistently at inference time.  Auto-regressive prediction feeds each
    one-step forecast back as the next input.
    """

    def __init__(
        self,
        seq_len:     int   = 144,
        hidden_size: int   = 64,
        num_layers:  int   = 2,
        lr:          float = 1e-3,
        batch_size:  int   = 128,
        epochs:      int   = 50,
        patience:    int   = 7,
        dropout:     float = 0.1,
        val_frac:    float = 0.10,
        verbose:     bool  = False,
        print_every: int   = 5,
        label:       str   = "",
    ):
        self.seq_len     = seq_len
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lr          = lr
        self.batch_size  = batch_size
        self.epochs      = epochs
        self.patience    = patience
        self.dropout     = dropout
        self.val_frac    = val_frac
        self.verbose     = verbose
        self.print_every = print_every
        self.label       = label

        self.device      = _best_device()
        self.model_      : Optional[_LSTMNet] = None
        self.mean_       : float = 0.0
        self.std_        : float = 1.0
        self.train_time_ : float = 0.0
        self.forecast_time_: float = 0.0
        self.history_    : dict = {"train_loss": [], "val_loss": []}
        self.best_epoch_ : int = 0

    def _norm(self, x):  return (x - self.mean_) / (self.std_ + 1e-8)
    def _denorm(self, x): return x * (self.std_ + 1e-8) + self.mean_

    def fit(self, train_series: np.ndarray) -> "LSTMForecaster":
        t0 = time.perf_counter()
        self.mean_ = float(np.mean(train_series))
        self.std_  = float(np.std(train_series))
        norm = self._norm(train_series.astype(np.float32))

        val_n = max(int(len(norm) * self.val_frac), self.seq_len + 1)
        tr_ds = _SlidingWindowDataset(norm[:-val_n], self.seq_len)
        va_ds = _SlidingWindowDataset(norm[-val_n:],  self.seq_len)
        tr_dl = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True,  drop_last=False)
        va_dl = DataLoader(va_ds, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.model_ = _LSTMNet(self.hidden_size, self.num_layers, self.dropout).to(self.device)
        n_params = _count_params(self.model_)
        opt   = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
        crit  = nn.MSELoss()

        if self.verbose:
            tag = f"  [{self.label}] " if self.label else "  "
            print(f"{tag}LSTM  H={self.hidden_size}  L={self.num_layers}  "
                  f"lr={self.lr:.0e}  device={self.device}  params={n_params:,}")
            print(f"  {'':>4}  Train windows: {len(tr_ds):,}   Val windows: {len(va_ds):,}   "
                  f"Batch: {self.batch_size}   Max epochs: {self.epochs}")
            print(f"  {'-'*72}")

        best_val, no_imp, best_state = float("inf"), 0, None
        stopped_ep = 0
        for ep in range(self.epochs):
            tr_loss = _run_epoch(self.model_, tr_dl, crit, self.device, opt)
            va_loss = _run_epoch(self.model_, va_dl, crit, self.device)
            cur_lr  = opt.param_groups[0]["lr"]
            sched.step(va_loss)
            self.history_["train_loss"].append(tr_loss)
            self.history_["val_loss"].append(va_loss)
            is_best = va_loss < best_val
            if is_best:
                best_val = va_loss
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                no_imp = 0
                self.best_epoch_ = ep + 1
            else:
                no_imp += 1
                if no_imp >= self.patience:
                    stopped_ep = ep + 1
                    break
            if self.verbose and ((ep + 1) % self.print_every == 0 or is_best):
                marker = ""
                print(f"  Ep {ep+1:>3}/{self.epochs}  "
                      f"train={tr_loss:.5f}  val={va_loss:.5f}  lr={cur_lr:.2e}{marker}")

        if best_state:
            self.model_.load_state_dict(best_state)
        self.train_time_ = time.perf_counter() - t0

        if self.verbose:
            stop_info = (f"early stop at ep {stopped_ep}" if stopped_ep
                         else f"completed {self.epochs} epochs")
            print(f"  {'-'*72}")
            print(f"  Done  |  {stop_info}  |  best ep: {self.best_epoch_}  "
                  f"|  best val MSE: {best_val:.5f}  |  time: {self.train_time_:.1f}s")
            print()
        return self

    def predict(self, context: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Pure auto-regressive rollout (multi-step ahead).
        Uses its own predictions as context — suffers from error accumulation
        on long horizons. Kept for failure-analysis demonstrations.
        """
        t0 = time.perf_counter()
        self.model_.eval()
        buf = self._norm(context[-self.seq_len:].astype(np.float32)).copy()
        preds = []
        with torch.no_grad():
            for _ in range(n_steps):
                xb   = torch.from_numpy(buf).unsqueeze(0).to(self.device)
                step = self.model_(xb).squeeze().item()
                preds.append(step)
                buf  = np.roll(buf, -1)
                buf[-1] = step
        self.forecast_time_ = time.perf_counter() - t0
        return self._denorm(np.array(preds, dtype=np.float32))

    def predict_rolling(self, train_series: np.ndarray, test_series: np.ndarray) -> np.ndarray:
        """
        Rolling one-step-ahead forecast (standard evaluation protocol).
        At each test step t, the model receives the actual previous seq_len
        observations as context and predicts exactly 1 step ahead.
        No error accumulation — directly comparable to Holt-Winters.
        """
        t0 = time.perf_counter()
        full = np.concatenate([train_series, test_series]).astype(np.float32)
        n_tr = len(train_series)
        self.model_.eval()
        preds = []
        with torch.no_grad():
            for t in range(len(test_series)):
                ctx_start = n_tr + t - self.seq_len
                ctx = self._norm(full[ctx_start : n_tr + t])
                xb  = torch.from_numpy(ctx).unsqueeze(0).to(self.device)
                step = self.model_(xb).squeeze().item()
                preds.append(float(self._denorm(np.float32(step))))
        self.forecast_time_ = time.perf_counter() - t0
        return np.clip(np.array(preds, dtype=np.float32), 0, None)

    def save(self, path: Path):
        torch.save({"state_dict": self.model_.state_dict(),
                    "mean": self.mean_, "std": self.std_,
                    "config": self._config()}, path)

    def _config(self) -> dict:
        return dict(seq_len=self.seq_len, hidden_size=self.hidden_size,
                    num_layers=self.num_layers, lr=self.lr,
                    batch_size=self.batch_size, epochs=self.epochs,
                    patience=self.patience, dropout=self.dropout,
                    val_frac=self.val_frac)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Encoder-only Transformer
# ══════════════════════════════════════════════════════════════════════════════

class _PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, :x.size(1), :])


class _TransformerNet(nn.Module):
    """
    Encoder-only Transformer for scalar time-series regression.

    Architecture
    ------------
    1. Linear input projection  : 1 → d_model  (per time step)
    2. Sinusoidal positional enc : injects temporal order
    3. Transformer encoder      : nhead-head self-attention × num_layers
    4. Regression head          : last token → d_model//2 → ReLU → 1
    """

    def __init__(self, seq_len: int, d_model: int, nhead: int,
                 num_layers: int, dim_ff: int, dropout: float):
        super().__init__()
        self.proj    = nn.Linear(1, d_model)
        self.pos_enc = _PositionalEncoding(d_model, max_len=seq_len + 4, dropout=dropout)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head    = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x.unsqueeze(-1))   # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)              # (B, T, d_model)
        return self.head(x[:, -1, :])    # (B, 1)


class TransformerForecaster:
    """Training wrapper for _TransformerNet (mirrors LSTMForecaster interface)."""

    def __init__(
        self,
        seq_len:    int   = 144,
        d_model:    int   = 64,
        nhead:      int   = 4,
        num_layers: int   = 2,
        dim_ff:     int   = 128,
        lr:         float = 1e-3,
        batch_size: int   = 128,
        epochs:     int   = 50,
        patience:   int   = 7,
        dropout:    float = 0.1,
        val_frac:   float = 0.10,
        verbose:    bool  = False,
        print_every:int   = 5,
        label:      str   = "",
    ):
        self.seq_len    = seq_len
        self.d_model    = d_model
        self.nhead      = nhead
        self.num_layers = num_layers
        self.dim_ff     = dim_ff
        self.lr         = lr
        self.batch_size = batch_size
        self.epochs     = epochs
        self.patience   = patience
        self.dropout    = dropout
        self.val_frac   = val_frac
        self.verbose    = verbose
        self.print_every = print_every
        self.label      = label

        self.device      = _best_device()
        self.model_      : Optional[_TransformerNet] = None
        self.mean_       : float = 0.0
        self.std_        : float = 1.0
        self.train_time_ : float = 0.0
        self.forecast_time_: float = 0.0
        self.history_    : dict = {"train_loss": [], "val_loss": []}
        self.best_epoch_ : int = 0

    def _norm(self, x):   return (x - self.mean_) / (self.std_ + 1e-8)
    def _denorm(self, x): return x * (self.std_ + 1e-8) + self.mean_

    def fit(self, train_series: np.ndarray) -> "TransformerForecaster":
        t0 = time.perf_counter()
        self.mean_ = float(np.mean(train_series))
        self.std_  = float(np.std(train_series))
        norm = self._norm(train_series.astype(np.float32))

        val_n = max(int(len(norm) * self.val_frac), self.seq_len + 1)
        tr_ds = _SlidingWindowDataset(norm[:-val_n], self.seq_len)
        va_ds = _SlidingWindowDataset(norm[-val_n:],  self.seq_len)
        tr_dl = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True,  drop_last=False)
        va_dl = DataLoader(va_ds, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.model_ = _TransformerNet(
            self.seq_len, self.d_model, self.nhead,
            self.num_layers, self.dim_ff, self.dropout,
        ).to(self.device)
        n_params = _count_params(self.model_)
        opt   = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
        crit  = nn.MSELoss()

        if self.verbose:
            tag = f"  [{self.label}] " if self.label else "  "
            print(f"{tag}Transformer  d={self.d_model}  nhead={self.nhead}  "
                  f"L={self.num_layers}  ff={self.dim_ff}  lr={self.lr:.0e}  "
                  f"device={self.device}  params={n_params:,}")
            print(f"  {'':>4}  Train windows: {len(tr_ds):,}   Val windows: {len(va_ds):,}   "
                  f"Batch: {self.batch_size}   Max epochs: {self.epochs}")
            print(f"  {'-'*72}")

        best_val, no_imp, best_state = float("inf"), 0, None
        stopped_ep = 0
        for ep in range(self.epochs):
            tr_loss = _run_epoch(self.model_, tr_dl, crit, self.device, opt)
            va_loss = _run_epoch(self.model_, va_dl, crit, self.device)
            cur_lr  = opt.param_groups[0]["lr"]
            sched.step(va_loss)
            self.history_["train_loss"].append(tr_loss)
            self.history_["val_loss"].append(va_loss)
            is_best = va_loss < best_val
            if is_best:
                best_val = va_loss
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                no_imp = 0
                self.best_epoch_ = ep + 1
            else:
                no_imp += 1
                if no_imp >= self.patience:
                    stopped_ep = ep + 1
                    break
            if self.verbose and ((ep + 1) % self.print_every == 0 or is_best):
                marker = ""
                print(f"  Ep {ep+1:>3}/{self.epochs}  "
                      f"train={tr_loss:.5f}  val={va_loss:.5f}  lr={cur_lr:.2e}{marker}")

        if best_state:
            self.model_.load_state_dict(best_state)
        self.train_time_ = time.perf_counter() - t0

        if self.verbose:
            stop_info = (f"early stop at ep {stopped_ep}" if stopped_ep
                         else f"completed {self.epochs} epochs")
            print(f"  {'-'*72}")
            print(f"  Done  |  {stop_info}  |  best ep: {self.best_epoch_}  "
                  f"|  best val MSE: {best_val:.5f}  |  time: {self.train_time_:.1f}s")
            print()
        return self

    def predict(self, context: np.ndarray, n_steps: int) -> np.ndarray:
        """Pure auto-regressive rollout — kept for failure-analysis demonstrations."""
        t0 = time.perf_counter()
        self.model_.eval()
        buf = self._norm(context[-self.seq_len:].astype(np.float32)).copy()
        preds = []
        with torch.no_grad():
            for _ in range(n_steps):
                xb   = torch.from_numpy(buf).unsqueeze(0).to(self.device)
                step = self.model_(xb).squeeze().item()
                preds.append(step)
                buf  = np.roll(buf, -1)
                buf[-1] = step
        self.forecast_time_ = time.perf_counter() - t0
        return self._denorm(np.array(preds, dtype=np.float32))

    def predict_rolling(self, train_series: np.ndarray, test_series: np.ndarray) -> np.ndarray:
        """
        Rolling one-step-ahead forecast (standard evaluation protocol).
        At each test step t the model receives the actual previous seq_len
        observations as context and predicts exactly 1 step ahead.
        """
        t0 = time.perf_counter()
        full = np.concatenate([train_series, test_series]).astype(np.float32)
        n_tr = len(train_series)
        self.model_.eval()
        preds = []
        with torch.no_grad():
            for t in range(len(test_series)):
                ctx_start = n_tr + t - self.seq_len
                ctx = self._norm(full[ctx_start : n_tr + t])
                xb  = torch.from_numpy(ctx).unsqueeze(0).to(self.device)
                step = self.model_(xb).squeeze().item()
                preds.append(float(self._denorm(np.float32(step))))
        self.forecast_time_ = time.perf_counter() - t0
        return np.clip(np.array(preds, dtype=np.float32), 0, None)

    def save(self, path: Path):
        torch.save({"state_dict": self.model_.state_dict(),
                    "mean": self.mean_, "std": self.std_,
                    "config": self._config()}, path)

    def _config(self) -> dict:
        return dict(seq_len=self.seq_len, d_model=self.d_model, nhead=self.nhead,
                    num_layers=self.num_layers, dim_ff=self.dim_ff, lr=self.lr,
                    batch_size=self.batch_size, epochs=self.epochs,
                    patience=self.patience, dropout=self.dropout,
                    val_frac=self.val_frac)
