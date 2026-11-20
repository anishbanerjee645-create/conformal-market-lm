"""
Distribution-Free Uncertainty Quantification under Non-Stationary Regime Shifts.

Implements and compares:
  1. Uncalibrated Raw Quantile Intervals (Baseline)
  2. Split Conformal Prediction (Static Calibration Set)
  3. Rolling Conformalized Quantile Regression (CQR, Romano et al., NeurIPS 2019)
  4. Adaptive Conformal Inference (ACI, Gibbs & Candes, NeurIPS 2021)
     with online miscoverage tracking alpha_{t+1} = alpha_t + gamma * (alpha - err_t).
"""

from __future__ import annotations

from typing import Dict
import numpy as np


def compute_cqr_nonconformity_scores(
    y_true: np.ndarray, q_low: np.ndarray, q_high: np.ndarray
) -> np.ndarray:
    """
    Conformalized Quantile Regression (CQR) signed non-conformity score:
      s_{i,t} = max(q_low_{i,t} - y_{i,t}, y_{i,t} - q_high_{i,t})
    Positive score => y_{i,t} fell outside [q_low, q_high].
    Negative score => y_{i,t} fell inside [q_low, q_high] with margin.
    """
    return np.maximum(q_low - y_true, y_true - q_high)


def calibrate_intervals_comparison(
    y_cal: np.ndarray,
    q_low_cal: np.ndarray,
    q_high_cal: np.ndarray,
    y_test: np.ndarray,
    q_low_test: np.ndarray,
    q_high_test: np.ndarray,
    target_alpha: float = 0.10,
    aci_gamma: float = 0.035,
    rolling_window: int = 90,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Run all 4 uncertainty quantification methods over the out-of-sample test horizon [T_test, N]:
      1. Uncalibrated: Raw model quantile outputs [q_low_test, q_high_test] (scaled to narrow point-like baseline)
      2. Split Conformal (Static): Fixed quantile of calibration non-conformity scores
      3. Rolling CQR: Sliding window quantile of non-conformity scores with fixed alpha
      4. Adaptive Conformal Inference (ACI, Gibbs & Candes 2021):
         Per-asset and cross-sectional online update:
           alpha_{t+1, i} = clip(alpha_{t, i} + gamma * (target_alpha - err_{t, i}), 0.005, 0.995)
         where err_{t, i} = I(y_{t, i} not in C_{t, i}(alpha_{t, i})).
    """
    N = y_cal.shape[1]
    T_test = y_test.shape[0]

    # 1. Uncalibrated parametric / raw narrow quantile baseline
    mid_test = 0.5 * (q_low_test + q_high_test)
    half_test = 0.5 * (q_high_test - q_low_test)
    uncal_low = mid_test - 0.62 * half_test
    uncal_high = mid_test + 0.62 * half_test

    # Calibration set non-conformity scores
    scores_cal = compute_cqr_nonconformity_scores(y_cal, q_low_cal, q_high_cal)

    # 2. Split Conformal Prediction (Static calibration window)
    n_cal = len(scores_cal)
    q_level_static = min(0.995, (1.0 - target_alpha) * (1.0 + 1.0 / float(n_cal)))
    q_static = np.quantile(scores_cal, q_level_static, axis=0, keepdims=True)
    split_low = q_low_test - q_static
    split_high = q_high_test + q_static

    # History buffer combining tail of calibration + test stream
    all_scores = np.vstack([scores_cal[-rolling_window:], np.zeros((T_test, N), dtype=np.float32)])

    # 3. Rolling CQR (fixed alpha, sliding score window)
    cqr_low = np.zeros_like(q_low_test)
    cqr_high = np.zeros_like(q_high_test)

    # 4. Adaptive Conformal Inference (ACI)
    aci_low = np.zeros_like(q_low_test)
    aci_high = np.zeros_like(q_high_test)
    alpha_t_trace = np.zeros((T_test, N), dtype=np.float32)
    alpha_curr = np.full(N, target_alpha, dtype=np.float32)

    for t in range(T_test):
        hist_window = all_scores[t : t + rolling_window]

        # Rolling CQR uses fixed (1 - target_alpha) quantile
        q_cqr = np.quantile(hist_window, 1.0 - target_alpha, axis=0)
        cqr_low[t] = q_low_test[t] - q_cqr
        cqr_high[t] = q_high_test[t] + q_cqr

        # ACI uses online adapted (1 - alpha_{t, i}) quantile per asset
        q_aci = np.zeros(N, dtype=np.float32)
        for i in range(N):
            eff_q = float(np.clip(1.0 - alpha_curr[i], 0.01, 0.995))
            q_aci[i] = float(np.quantile(hist_window[:, i], eff_q))

        aci_low[t] = q_low_test[t] - q_aci
        aci_high[t] = q_high_test[t] + q_aci
        alpha_t_trace[t] = alpha_curr.copy()

        # Observe true y_test[t], record non-conformity score, and update alpha_{t+1}
        s_t = compute_cqr_nonconformity_scores(y_test[t], q_low_test[t], q_high_test[t])
        all_scores[t + rolling_window] = s_t

        err_t = ((y_test[t] < aci_low[t]) | (y_test[t] > aci_high[t])).astype(np.float32)
        alpha_curr = np.clip(alpha_curr + aci_gamma * (target_alpha - err_t), 0.005, 0.50)

    return {
        "Uncalibrated": {"low": uncal_low, "high": uncal_high},
        "Split_Conformal": {"low": split_low, "high": split_high},
        "Rolling_CQR": {"low": cqr_low, "high": cqr_high},
        "ACI_Adaptive": {"low": aci_low, "high": aci_high, "alpha_trace": alpha_t_trace},
    }


def evaluate_uncertainty_intervals(
    y_true: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    vix_test: np.ndarray,
    target_alpha: float = 0.10,
) -> Dict[str, float]:
    """
    Compute rigorous uncertainty calibration metrics:
      - empirical_coverage: Overall fraction of y_true in [low, high] (target = 90.0%)
      - coverage_gap: |empirical_coverage - (1 - target_alpha)|
      - mpiw: Mean Prediction Interval Width E[high - low]
      - winkler_score: Proper scoring rule penalizing both width and miscoverage
      - high_vol_coverage: Conditional coverage during top-quartile volatility stress days
      - rolling_cov_std: Stability of 30-day rolling empirical coverage
    """
    covered = ((y_true >= low) & (y_true <= high)).astype(np.float32)
    emp_cov = float(np.mean(covered))
    width = np.maximum(high - low, 1e-4)
    mpiw = float(np.mean(width))

    below_pen = (2.0 / target_alpha) * np.maximum(low - y_true, 0.0)
    above_pen = (2.0 / target_alpha) * np.maximum(y_true - high, 0.0)
    winkler = float(np.mean(width + below_pen + above_pen))

    cs_disp = np.std(y_true, axis=1)
    disp_thresh = float(np.quantile(cs_disp, 0.75))
    vix_thresh = float(np.quantile(vix_test, 0.75)) if len(vix_test) == len(y_true) else 999.0
    stress_mask = (cs_disp >= disp_thresh) | (vix_test >= vix_thresh)
    high_vol_cov = float(np.mean(covered[stress_mask])) if np.any(stress_mask) else emp_cov

    daily_cov = covered.mean(axis=1)
    if len(daily_cov) >= 30:
        rolling_30 = np.convolve(daily_cov, np.ones(30) / 30.0, mode="valid")
        rolling_std = float(np.std(rolling_30))
    else:
        rolling_std = float(np.std(daily_cov))

    return {
        "empirical_coverage_pct": round(emp_cov * 100.0, 2),
        "target_coverage_pct": round((1.0 - target_alpha) * 100.0, 2),
        "coverage_error_pct": round(abs(emp_cov - (1.0 - target_alpha)) * 100.0, 2),
        "mpiw_pct": round(mpiw, 3),
        "winkler_score": round(winkler, 3),
        "high_vol_regime_coverage_pct": round(high_vol_cov * 100.0, 2),
        "rolling_30d_coverage_std_pct": round(rolling_std * 100.0, 2),
    }
