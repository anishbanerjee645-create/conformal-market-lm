"""
Bulk self-supervised training on EVERY ticker in the Stock_Predictor dataset
(26YRS_closes_*.csv: ~2,247 NSE + ~2,443 BSE series, 2000-2026).

Trains the online self-supervised learner (next-bar prediction + masked-feature
denoising) on the whole panel, validates on a held-out final window, and saves
the weights as the starting state for live continual learning.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.self_supervised import FEATURE_DIM, OnlineSelfSupervisedLearner  # noqa: E402


def find_closes_csv() -> Path:
    matches = sorted(BASE.rglob("26YRS_closes_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No 26YRS_closes_*.csv dataset found under {BASE}")
    return matches[-1]


def _as_df(arr: np.ndarray, ref: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(arr, index=ref.index, columns=ref.columns)


def panel_features(close: pd.DataFrame) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    c = close.where(close > 0)
    r1 = _as_df(np.tanh(((c / c.shift(1) - 1.0) * 100.0 / 2.5).to_numpy(dtype=np.float64)), c)
    r5 = _as_df(np.tanh(((c / c.shift(5) - 1.0) * 100.0 / 4.5).to_numpy(dtype=np.float64)), c)
    r20 = _as_df(np.tanh(((c / c.shift(20) - 1.0) * 100.0 / 8.0).to_numpy(dtype=np.float64)), c)
    d = c.diff()
    g = d.clip(lower=0.0).rolling(14).mean()
    losses = (-d).clip(lower=0.0).rolling(14).mean()
    rsi = 100.0 - 100.0 / (1.0 + g / (losses + 1e-6))
    rsi_n = (rsi - 50.0) / 25.0
    macd_raw = ((c.rolling(6).mean() - c.rolling(20).mean()) / c) * 100.0 / 2.0
    macd = _as_df(np.tanh(macd_raw.to_numpy(dtype=np.float64)), c)
    log_c = _as_df(np.log(c.to_numpy(dtype=np.float64)), c)
    lr = log_c.diff() * 100.0
    vr = (lr.rolling(10).std() / 1.2).clip(upper=2.5) - 1.0
    nxt = ((c.shift(-1) / c - 1.0) * 100.0).clip(-8.0, 8.0)
    return [r1, r5, r20, rsi_n, macd, vr], nxt


def main() -> None:
    t0 = time.time()
    csv_path = find_closes_csv()
    df = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date").sort_index()
    df = df.dropna(axis=1, thresh=250)
    df = df.where(df > 0)
    print(f"[load] {df.shape[1]} tickers x {df.shape[0]} days from {csv_path.name}")

    if len(df) <= 504:
        raise ValueError(f"Dataset too short ({len(df)} rows); need > 504 trading days.")
    dates = pd.DatetimeIndex(df.index)
    cut = dates[-504]  # hold out the last ~2 years
    feats, nxt = panel_features(df)

    def stack(mask_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cols = [f.loc[mask_rows].to_numpy(dtype=np.float64).ravel() for f in feats]
        y = nxt.loc[mask_rows].to_numpy(dtype=np.float64).ravel()
        X = np.column_stack(cols + [np.ones_like(y)])
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        return X[ok], y[ok]

    Xtr, ytr = stack(np.asarray(dates < cut))
    Xva, yva = stack(np.asarray(dates >= cut))
    # Winsorise to avoid corporate-action artefacts (splits/bonus) dominating
    keep = np.abs(ytr) < 8.0
    Xtr, ytr = Xtr[keep], ytr[keep]
    print(f"[data] train samples={len(ytr):,}  validation samples={len(yva):,}")

    L = OnlineSelfSupervisedLearner()
    rng = np.random.default_rng(7)
    delta = 1.5
    n = len(ytr)
    bs = 4096
    for epoch in range(4):
        perm = rng.permutation(n)
        for s in range(0, n, bs):
            idx = perm[s : s + bs]
            Xb, yb = Xtr[idx], ytr[idx]
            m = rng.random((len(idx), FEATURE_DIM - 1)) > 0.20  # masked denoising
            Xm = Xb.copy()
            Xm[:, : FEATURE_DIM - 1] *= m
            e = Xm @ L.weights - yb
            gr = np.where(np.abs(e) <= delta, e, delta * np.sign(e))
            L._adam_step(Xm.T @ gr / len(idx), lr=0.002)
        pv = Xva @ L.weights
        hit = float(np.mean((pv >= 0) == (yva >= 0)))
        print(f"[epoch {epoch+1}] val MAE={np.mean(np.abs(pv-yva)):.4f}  directional acc={hit:.4f}")

    base_hit = float(np.mean(yva >= 0))
    pv = Xva @ L.weights
    hit = float(np.mean((pv >= 0) == (yva >= 0)))
    ic = (
        float(np.corrcoef(pv, yva)[0, 1])
        if len(pv) > 1 and float(np.std(pv)) > 1e-12 and float(np.std(yva)) > 1e-12
        else 0.0
    )

    # Seed replay buffer with a broad sample so live learning never forgets history
    ridx = rng.choice(n, size=min(L.max_replay_size, n), replace=False)
    L.replay_buffer = np.column_stack([Xtr[ridx], ytr[ridx]]).tolist()
    L.total_bars_learned += n
    L.total_ssl_steps += 4 * (n // bs + 1)
    L.directional_hits, L.directional_total = int(hit * len(yva)), len(yva)
    L.realized_mae_ema = float(np.mean(np.abs(pv - yva)))
    L.last_learned_symbol = "ALL-TICKERS"
    L._save_state()

    rep = {
        "tickers": int(df.shape[1]),
        "train_samples": int(n),
        "val_samples": int(len(yva)),
        "val_directional_accuracy": hit,
        "val_base_rate_up": base_hit,
        "val_information_coefficient": ic,
        "weights": L.weights.tolist(),
        "seconds": round(time.time() - t0, 1),
    }
    reports_dir = BASE / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "bulk_training_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
