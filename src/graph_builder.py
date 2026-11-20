"""
Dynamic Multi-Relational Graph Construction for Cross-Industry Contagion Modeling.
Constructs time-varying edge features E_t in R^{N x N x D_e} combining:
  1. Intra-Sector & Macroeconomic Supply-Chain Structural Prior (A_prior)
  2. Exponentially-Weighted Rolling Return Correlation Graph (A_corr,t)
  3. Asymmetric Tail-Risk Co-Exceedance & Lead-Lag Contagion Graph (A_tail,t)
"""

from __future__ import annotations

from typing import Dict, List, Tuple, TypedDict
import numpy as np


class ContagionGraphOutput(TypedDict):
    E_tensor: np.ndarray
    sector_contagion: np.ndarray
    unique_sectors: List[str]
    A_intra: np.ndarray
    A_supply: np.ndarray


CROSS_SECTOR_COUPLING: Dict[Tuple[str, str], float] = {
    ("Energy", "Auto_Ind"): 0.65,
    ("Energy", "FMCG"): 0.50,
    ("Energy", "Metals"): 0.62,
    ("Metals", "Auto_Ind"): 0.68,
    ("Metals", "Banking"): 0.48,
    ("Banking", "Auto_Ind"): 0.70,
    ("Banking", "IT"): 0.45,
    ("IT", "Pharma"): 0.55,
    ("FMCG", "Pharma"): 0.40,
}


def build_structural_prior_matrix(tickers: List[str], sectors: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build static structural adjacency matrices:
      - A_intra [N, N]: 1.0 if stocks i, j belong to the same sector (plus self-loops), 0 otherwise
      - A_supply [N, N]: prior cross-sector supply-chain / macro coupling weight
    """
    N = len(tickers)
    A_intra = np.eye(N, dtype=np.float32)
    A_supply = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            sec_i, sec_j = sectors[i], sectors[j]
            if sec_i == sec_j:
                A_intra[i, j] = 1.0
            else:
                w = CROSS_SECTOR_COUPLING.get(
                    (sec_i, sec_j),
                    CROSS_SECTOR_COUPLING.get((sec_j, sec_i), 0.15),
                )
                A_supply[i, j] = w

    A_intra_norm = A_intra / (A_intra.sum(axis=1, keepdims=True) + 1e-8)
    A_supply_norm = A_supply / (A_supply.sum(axis=1, keepdims=True) + 1e-8)
    return A_intra_norm, A_supply_norm


def compute_dynamic_contagion_graphs(
    returns_1d: np.ndarray,
    sectors: List[str],
    tickers: List[str],
    window: int = 45,
    corr_threshold: float = 0.25,
) -> ContagionGraphOutput:
    """
    Compute dynamic spatio-temporal edge tensors across all time steps t in [0, T-1]:
      - E_tensor [T, N, N, 4]
      - sector_contagion [T, S, S]
    """
    T, N = returns_1d.shape
    A_intra, A_supply = build_structural_prior_matrix(tickers, sectors)

    E_tensor = np.zeros((T, N, N, 4), dtype=np.float32)
    E_tensor[:, :, :, 0] = A_intra[None, :, :]
    E_tensor[:, :, :, 1] = A_supply[None, :, :]

    decay = np.exp(np.linspace(-1.5, 0.0, window)).astype(np.float32)
    decay /= decay.sum()

    unique_sectors: List[str] = list(dict.fromkeys(sectors))
    S = len(unique_sectors)
    sector_indices: Dict[str, List[int]] = {
        sec: [idx for idx, s in enumerate(sectors) if s == sec]
        for sec in unique_sectors
    }
    sector_contagion = np.zeros((T, S, S), dtype=np.float32)

    for t in range(T):
        start_idx = max(0, t - window + 1)
        sub_ret = returns_1d[start_idx : t + 1]
        w_len = len(sub_ret)

        if w_len < 8:
            E_tensor[t, :, :, 2] = A_intra
            E_tensor[t, :, :, 3] = A_supply
            continue

        w_vec = decay[-w_len:][:, None]
        mean_r = (sub_ret * w_vec).sum(axis=0, keepdims=True)
        centered = sub_ret - mean_r
        cov = (centered * w_vec).T @ centered
        std = np.sqrt(np.clip(np.diag(cov), 1e-8, None))
        corr = cov / (std[:, None] * std[None, :] + 1e-8)
        corr = np.clip(corr, -1.0, 1.0)

        mask = (np.abs(corr) >= corr_threshold).astype(np.float32)
        np.fill_diagonal(mask, 1.0)
        A_corr = np.abs(corr) * mask
        A_corr /= A_corr.sum(axis=1, keepdims=True) + 1e-8
        E_tensor[t, :, :, 2] = A_corr

        r_prev = sub_ret[:-1]
        r_curr = sub_ret[1:]

        down_prev = np.minimum(r_prev, 0.0)
        down_curr = np.minimum(r_curr, 0.0)

        norm_prev = np.sqrt((down_prev ** 2).sum(axis=0) + 1e-6)
        norm_curr = np.sqrt((down_curr ** 2).sum(axis=0) + 1e-6)
        tail_lead_lag = (down_curr.T @ down_prev) / (norm_curr[:, None] * norm_prev[None, :] + 1e-6)
        tail_lead_lag = np.clip(tail_lead_lag, 0.0, 1.0)
        np.fill_diagonal(tail_lead_lag, 0.5)
        tail_lead_lag /= tail_lead_lag.sum(axis=1, keepdims=True) + 1e-8
        E_tensor[t, :, :, 3] = tail_lead_lag

        combined_adj = 0.5 * A_corr + 0.5 * tail_lead_lag
        for s_target_idx, s_target in enumerate(unique_sectors):
            idx_i = sector_indices[s_target]
            for s_source_idx, s_source in enumerate(unique_sectors):
                idx_j = sector_indices[s_source]
                sector_contagion[t, s_target_idx, s_source_idx] = float(
                    combined_adj[np.ix_(idx_i, idx_j)].mean()
                )

    return {
        "E_tensor": E_tensor,
        "sector_contagion": sector_contagion,
        "unique_sectors": unique_sectors,
        "A_intra": A_intra,
        "A_supply": A_supply,
    }
