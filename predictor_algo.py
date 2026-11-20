"""
ConsensusSignal: end-to-end stock forecasting + decision algorithm.

Pipeline (per day, per stock)
  1. Tokenise the last 64 days into vol-normalised return tokens.
  2. MarketLM (transformer) -> distribution over next token
        -> E[z], P(up), entropy (uncertainty).
  3. Online linear learner (7 causal features) -> r_hat.
  4. Fuse: s = 0.5*xs_z(E[z]_LM) + 0.5*xs_z(r_hat / sigma)       (xs_z = cross-sectional z-score)
  5. Confidence gate: drop names whose LM entropy is in the least-certain 30%.
  6. Portfolio: long top decile, short bottom decile of gated scores (market-neutral),
        weights ~ 1/sigma, gross exposure 100%, 10 bps cost per unit turnover.
  7. Online ACI-style safety: if rolling 60-day realised hit-rate < 50% the book
        is cut to half size until it recovers (a model-failure circuit-breaker).
Run:  python Stock_Predictor/predictor_algo.py  (walk-forward on the held-out last ~2 years)
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.market_lm import CTX, K, load_market_lm, tokenise  # noqa: E402
from src.self_supervised import OnlineSelfSupervisedLearner  # noqa: E402
from train_all_data import panel_features  # noqa: E402

BASE = Path(__file__).resolve().parent
COST_BPS = 10.0
HOLD_DAYS = 504


def xs_z(a: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score per row without NaN slice warnings."""
    out = np.full_like(a, np.nan, dtype=np.float64)
    finite = np.isfinite(a)
    counts = finite.sum(axis=1)
    valid_rows = counts >= 2
    if not np.any(valid_rows):
        return out
    sub = a[valid_rows]
    m = np.nanmean(sub, axis=1, keepdims=True)
    s = np.nanstd(sub, axis=1, keepdims=True) + 1e-9
    out[valid_rows] = (sub - m) / s
    return out


def main(n_stocks: int = 300) -> None:
    t0 = time.time()
    bundle = load_market_lm()
    if bundle is None:
        raise FileNotFoundError(
            f"MarketLM checkpoint not found at {BASE / 'artifacts' / 'market_lm.pt'}. "
            "Run train_market_lm.py first."
        )
    model, edges, centers, _ = bundle
    learner = OnlineSelfSupervisedLearner()
    w = learner.weights.copy()

    csv_files = sorted(BASE.rglob("26YRS_closes_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No 26YRS_closes_*.csv dataset found under {BASE}")
    csv = csv_files[-1]

    df = pd.read_csv(csv, parse_dates=["Date"]).set_index("Date").sort_index()
    df = df.where(df > 0)
    T = len(df)
    start = max(CTX + 40, T - HOLD_DAYS)
    full = df.iloc[start - CTX - 40 :].dropna(axis=1)  # complete history over test span
    if full.shape[1] == 0:
        raise RuntimeError("No tickers have complete price history over the test window.")

    # Deterministically sample up to n_stocks tickers with complete history
    rng = np.random.default_rng(0)
    cols = [str(c) for c in rng.choice(full.columns.to_numpy(), size=min(n_stocks, full.shape[1]), replace=False)]
    px = df[cols].iloc[start - CTX - 40 :]
    dates = pd.DatetimeIndex(px.index[CTX + 40 :])
    N = len(cols)
    start_date = pd.Timestamp(dates[0]).date()
    end_date = pd.Timestamp(dates[-1]).date()
    print(f"universe {N} stocks, test days {len(dates)} ({start_date}..{end_date})", flush=True)

    # --- tokens & sigma ---
    Z = np.column_stack([tokenise(px[c].to_numpy(dtype=np.float64))[0] for c in cols])
    Z[~np.isfinite(Z)] = 0.0
    Z[np.abs(Z) >= 6.0] = 0.0
    TOK = np.clip(np.digitize(Z, edges), 0, K - 1)  # [Tn, N]

    log_ret = np.log(px / px.shift(1)).to_numpy(dtype=np.float64) * 100.0
    simple_ret = (px.pct_change() * 100.0).to_numpy(dtype=np.float64)
    sigma = pd.DataFrame(log_ret).rolling(20).std().to_numpy(dtype=np.float64)  # causal at decision time t

    # --- online learner features (aligned with px) ---
    feats, _ = panel_features(px)
    Fm = np.stack([f.to_numpy(dtype=np.float64) for f in feats] + [np.ones(px.shape, dtype=np.float64)], axis=-1)
    r_hat = Fm @ w

    Tn = len(px)
    first = CTX + 40
    ez = np.full((Tn, N), np.nan, dtype=np.float64)
    ent = np.full((Tn, N), np.nan, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for t in range(first, Tn - 1):  # decision at close of t, earn r_{t+1}
            win = torch.tensor(TOK[t - CTX + 1 : t + 1].T, dtype=torch.long)  # [N, CTX]
            p = F.softmax(model(win)[:, -1], dim=-1).numpy()
            ez[t] = p @ centers
            ent[t] = -(p * np.log2(p + 1e-12)).sum(axis=1)
    print(f"inference done {time.time()-t0:.0f}s", flush=True)

    sig = np.where(np.isfinite(sigma) & (sigma > 0.3), sigma, 0.3)
    lm_only = xs_z(ez)
    ol_only = xs_z(r_hat / sig)
    score = 0.5 * lm_only + 0.5 * ol_only

    # Use simple arithmetic next-day returns for portfolio PnL; drop |r| > 25% as split/data errors
    nextret = np.vstack([simple_ret[1:], np.full((1, N), np.nan, dtype=np.float64)])
    nextret[np.abs(nextret) > 25.0] = np.nan

    def backtest(sc: np.ndarray, gate: bool, breaker: bool) -> dict[str, float | int]:
        prev_w = np.zeros(N, dtype=np.float64)
        pnl_list: list[float] = []
        hits: list[float] = []
        ics: list[float] = []
        scale = 1.0
        for t in range(first, Tn - 1):
            s = sc[t].copy()
            if gate:
                ent_t = ent[t]
                if np.isfinite(ent_t).any():
                    thr = float(np.nanquantile(ent_t, 0.70))
                    s[ent_t > thr] = np.nan
            ok = np.isfinite(s) & np.isfinite(nextret[t])
            wv = np.zeros(N, dtype=np.float64)
            if int(ok.sum()) >= 30:
                s_ok = s[ok]
                r_ok = nextret[t][ok]
                lo, hi = np.nanquantile(s_ok, [0.1, 0.9])
                if hi > lo:
                    longs = ok & (s >= hi)
                    shorts = ok & (s <= lo) & ~longs
                    inv = 1.0 / sig[t]
                    if longs.any():
                        wv[longs] = inv[longs] / inv[longs].sum() * 0.5
                    if shorts.any():
                        wv[shorts] = -inv[shorts] / inv[shorts].sum() * 0.5
                if float(np.nanstd(s_ok)) > 1e-12 and float(np.nanstd(r_ok)) > 1e-12:
                    ic_val = pd.Series(s_ok).corr(pd.Series(r_ok), method="spearman")
                    if np.isfinite(ic_val):
                        ics.append(float(ic_val))
            if breaker and len(hits) >= 60:
                scale = 0.5 if float(np.mean(hits[-60:])) < 0.5 else 1.0
            wv *= scale
            gross = float(np.nansum(wv * np.nan_to_num(nextret[t]))) / 100.0
            cost = float(np.abs(wv - prev_w).sum()) * COST_BPS / 1e4
            pnl_list.append(gross - cost)
            hits.append(float(gross > 0.0))
            prev_w = wv

        pnl = np.asarray(pnl_list, dtype=np.float64)
        if pnl.size == 0:
            return {
                "total_return_pct": 0.0,
                "ann_sharpe_net": 0.0,
                "max_drawdown_pct": 0.0,
                "daily_hit_rate": 0.0,
                "mean_daily_rank_ic": 0.0,
                "days": 0,
            }
        eq = np.cumprod(1.0 + pnl)
        dd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
        return {
            "total_return_pct": float((eq[-1] - 1.0) * 100.0),
            "ann_sharpe_net": float(pnl.mean() / (pnl.std() + 1e-12) * np.sqrt(252.0)),
            "max_drawdown_pct": dd * 100.0,
            "daily_hit_rate": float((pnl > 0.0).mean()),
            "mean_daily_rank_ic": float(np.nanmean(ics)) if ics else 0.0,
            "days": int(pnl.size),
        }

    res = {
        "Fused + gate + breaker (ConsensusSignal)": backtest(score, True, True),
        "Fused only": backtest(score, False, False),
        "MarketLM only": backtest(lm_only, False, False),
        "Online learner only": backtest(ol_only, False, False),
        "Random scores (sanity check)": backtest(np.random.default_rng(1).normal(size=score.shape), False, False),
    }
    out = {
        "universe": N,
        "cost_bps_per_side": COST_BPS,
        "period": [str(start_date), str(end_date)],
        "results": res,
        "seconds": round(time.time() - t0, 1),
    }
    reports_dir = BASE / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "algo_backtest.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
