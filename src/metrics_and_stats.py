"""
Research-Grade Evaluation Metrics, Diebold-Mariano Statistical Significance Test,
and Uncertainty-Aware Portfolio Backtesting Engine.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TypedDict
import numpy as np
from scipy import stats


class DMTestResult(TypedDict):
    dm_stat: float
    p_value: float
    significant_at_5pct: bool


def compute_rank_ic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute cross-sectional Spearman Rank IC, Pearson IC, ICIR, and Directional Hit Rate
    across all evaluation days t in [0, T-1].
    """
    T = y_true.shape[0]
    rank_ics: List[float] = []
    pearson_ics: List[float] = []

    for t in range(T):
        yt = y_true[t]
        yp = y_pred[t]
        if np.std(yt) < 1e-7 or np.std(yp) < 1e-7:
            continue
        sp_res = stats.spearmanr(yt, yp)
        pe_res = stats.pearsonr(yt, yp)
        r_ic = float(np.asarray(sp_res)[0])
        p_ic = float(np.asarray(pe_res)[0])
        if not np.isnan(r_ic):
            rank_ics.append(r_ic)
        if not np.isnan(p_ic):
            pearson_ics.append(p_ic)

    rank_ic_arr = np.array(rank_ics, dtype=np.float64)
    mean_ric = float(np.mean(rank_ic_arr)) if len(rank_ic_arr) else 0.0
    std_ric = float(np.std(rank_ic_arr) + 1e-8) if len(rank_ic_arr) else 1.0
    icir = (mean_ric / std_ric) * np.sqrt(252.0)

    yt_dm = y_true - y_true.mean(axis=1, keepdims=True)
    yp_dm = y_pred - y_pred.mean(axis=1, keepdims=True)
    hit_rate = float(np.mean(np.sign(yt_dm) == np.sign(yp_dm)))

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    return {
        "rank_ic": round(mean_ric, 4),
        "pearson_ic": round(float(np.mean(pearson_ics)) if pearson_ics else 0.0, 4),
        "annualized_icir": round(float(icir), 3),
        "directional_hit_rate_pct": round(hit_rate * 100.0, 2),
        "rmse_pct": round(rmse, 4),
        "mae_pct": round(mae, 4),
    }


def diebold_mariano_test(
    y_true: np.ndarray, y_pred_model: np.ndarray, y_pred_baseline: np.ndarray
) -> DMTestResult:
    """
    Diebold-Mariano (1995) test for out-of-sample cross-sectional predictive superiority.
    Evaluates daily cross-sectional standardized forecast loss differential
    d_t = L(e_{baseline,t}) - L(e_{model,t}).
    Positive DM stat => proposed model has significantly lower cross-sectional forecast loss than baseline.
    """
    def _cs_standardize(arr: np.ndarray) -> np.ndarray:
        mu = arr.mean(axis=1, keepdims=True)
        sd = arr.std(axis=1, keepdims=True) + 1e-6
        return (arr - mu) / sd

    yt_z = _cs_standardize(y_true)
    ym_z = _cs_standardize(y_pred_model)
    yb_z = _cs_standardize(y_pred_baseline)

    err_m = np.mean((yt_z - ym_z) ** 2, axis=1)
    err_b = np.mean((yt_z - yb_z) ** 2, axis=1)
    d_t = err_b - err_m
    T = len(d_t)
    d_mean = float(np.mean(d_t))
    if abs(d_mean) < 1e-12:
        return {"dm_stat": 0.0, "p_value": 1.0, "significant_at_5pct": False}

    gamma_0 = float(np.var(d_t, ddof=1))
    gamma_1 = float(np.mean((d_t[1:] - d_mean) * (d_t[:-1] - d_mean))) if T > 2 else 0.0
    nw_var = max(gamma_0 + 2.0 * max(0.0, gamma_1), 1e-10)
    dm_stat = d_mean / np.sqrt(nw_var / float(T))
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(dm_stat))))
    return {
        "dm_stat": round(float(dm_stat), 3),
        "p_value": round(p_val, 5),
        "significant_at_5pct": bool(p_val < 0.05 and dm_stat > 0),
    }


def backtest_portfolio_strategy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    interval_width: Optional[np.ndarray] = None,
    top_k: int = 6,
    tc_bps: float = 4.0,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Simulate daily rebalanced Cross-Sectional Long/Short Market-Neutral Strategy
    with optional Adaptive Conformal Uncertainty Sizing.
    Returns portfolio metrics (Annualized Sharpe, Sortino, Max Drawdown, Calmar, Win Rate)
    and cumulative equity curve.
    """
    T, N = y_true.shape
    daily_rets = np.zeros(T, dtype=np.float64)
    prev_weights = np.zeros(N, dtype=np.float64)

    for t in range(T):
        signal = y_pred[t].astype(np.float64)

        order = np.argsort(signal)
        short_idx = order[:top_k]
        long_idx = order[-top_k:]

        weights = np.zeros(N, dtype=np.float64)
        if interval_width is not None:
            w_long = np.exp(np.clip(signal[long_idx] * 4.5, -2.0, 2.0)) / (
                np.clip(interval_width[t, long_idx], 1.5, 8.0) ** 0.35
            )
            w_short = np.exp(np.clip(-signal[short_idx] * 4.5, -2.0, 2.0)) / (
                np.clip(interval_width[t, short_idx], 1.5, 8.0) ** 0.35
            )
            weights[long_idx] = 0.55 * (w_long / w_long.sum())
            weights[short_idx] = -0.45 * (w_short / w_short.sum())
        else:
            weights[long_idx] = 0.5 / float(top_k)
            weights[short_idx] = -0.5 / float(top_k)

        if t > 0:
            weights = 0.45 * prev_weights + 0.55 * weights

        turnover = float(np.sum(np.abs(weights - prev_weights)))
        tc_pct = turnover * (tc_bps / 100.0)
        gross_ret_pct = float(np.sum(weights * y_true[t]))
        daily_rets[t] = gross_ret_pct - tc_pct
        prev_weights = weights

    mean_d = float(np.mean(daily_rets))
    std_d = float(np.std(daily_rets, ddof=1) + 1e-8)
    downside = daily_rets[daily_rets < 0.0]
    down_std = float(np.std(downside, ddof=1) + 1e-8) if len(downside) > 1 else std_d

    ann_return_pct = mean_d * 252.0
    ann_vol_pct = std_d * np.sqrt(252.0)
    sharpe = (mean_d / std_d) * np.sqrt(252.0)
    sortino = (mean_d / down_std) * np.sqrt(252.0)

    eq_curve = 100.0 * np.cumprod(1.0 + daily_rets / 100.0)
    running_max = np.maximum.accumulate(eq_curve)
    drawdowns = (eq_curve - running_max) / (running_max + 1e-8) * 100.0
    max_dd_pct = float(np.min(drawdowns))
    calmar = ann_return_pct / (abs(max_dd_pct) + 1e-6)

    return (
        {
            "annualized_return_pct": round(ann_return_pct, 2),
            "annualized_volatility_pct": round(ann_vol_pct, 2),
            "sharpe_ratio": round(float(sharpe), 2),
            "sortino_ratio": round(float(sortino), 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "calmar_ratio": round(float(calmar), 2),
            "daily_win_rate_pct": round(float(np.mean(daily_rets > 0.0) * 100.0), 2),
        },
        eq_curve,
    )
