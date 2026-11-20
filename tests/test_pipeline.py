"""
Automated Verification & Mathematical Property Tests for ST-GATv2 + ACI Pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from src.conformal import calibrate_intervals_comparison, evaluate_uncertainty_intervals
from src.graph_builder import build_structural_prior_matrix, compute_dynamic_contagion_graphs
from src.models import SpatioTemporalGAT, STGATConfig


def test_all() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    dataset_path = base_dir / "data" / "processed" / "st_gat_dataset.npz"
    assert dataset_path.exists(), "Processed dataset missing!"

    data = np.load(dataset_path, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    Y = np.asarray(data["Y"], dtype=np.float32)
    tickers: list[str] = [str(t) for t in data["tickers"].tolist()]
    sectors: list[str] = [str(s) for s in data["sectors"].tolist()]

    assert X.ndim == 3 and Y.ndim == 2, "Invalid X/Y dimensions"
    assert X.shape[0] == Y.shape[0] and X.shape[1] == Y.shape[1], "Mismatched T or N"
    assert not np.isnan(X).any() and not np.isnan(Y).any(), "NaNs detected in dataset"
    print(f"[PASS] Dataset integrity verified: X={X.shape}, Y={Y.shape}")

    # Test 2: Graph builder row-normalization & shapes
    A_intra, A_supply = build_structural_prior_matrix(tickers, sectors)
    np.testing.assert_allclose(A_intra.sum(axis=1), np.ones(len(tickers)), atol=1e-4)
    np.testing.assert_allclose(A_supply.sum(axis=1), np.ones(len(tickers)), atol=1e-4)

    g_out = compute_dynamic_contagion_graphs(X[:80, :, 0], sectors, tickers, window=20)
    E_sub = g_out["E_tensor"]
    assert E_sub.shape == (80, len(tickers), len(tickers), 4)
    print(f"[PASS] Dynamic Contagion Graph tensor verified: E={E_sub.shape}")

    # Test 3: ST-GATv2 forward & backward pass shape and quantile ordering (q_low < y_hat < q_high)
    cfg = STGATConfig(in_features=X.shape[2], hidden_dim=16, num_heads=4, epochs=3)
    model = SpatioTemporalGAT(cfg)
    losses = model.fit(X[:60], Y[:60], E_sub[:60], batch_size=32)
    assert len(losses) == 3 and np.isfinite(losses[-1])
    preds = model.predict(X[60:80], E_sub[60:80])
    assert np.all(preds["q_low"] < preds["y_hat"]) and np.all(preds["y_hat"] < preds["q_high"])
    print(f"[PASS] ST-GATv2 quantile monotonicity verified (final loss={losses[-1]:.4f})")

    # Test 4: Adaptive Conformal Inference coverage tracking
    cal_res = calibrate_intervals_comparison(
        y_cal=Y[:40],
        q_low_cal=preds["q_low"][:20].repeat(2, axis=0),
        q_high_cal=preds["q_high"][:20].repeat(2, axis=0),
        y_test=Y[60:80],
        q_low_test=preds["q_low"],
        q_high_test=preds["q_high"],
        target_alpha=0.10,
        aci_gamma=0.05,
        rolling_window=20,
    )
    assert "ACI_Adaptive" in cal_res
    assert cal_res["ACI_Adaptive"]["low"].shape == (20, len(tickers))
    eval_metrics = evaluate_uncertainty_intervals(
        y_true=Y[60:80],
        low=cal_res["ACI_Adaptive"]["low"],
        high=cal_res["ACI_Adaptive"]["high"],
        vix_test=X[60:80, 0, 9],
        target_alpha=0.10,
    )
    assert "empirical_coverage_pct" in eval_metrics
    print("[PASS] Adaptive Conformal Inference (ACI) online calibration verified.")

    # Test 5: Check experimental_results.json if generated
    res_file = base_dir / "reports" / "experimental_results.json"
    if res_file.exists():
        res_json = json.loads(res_file.read_text(encoding="utf-8"))
        assert "ST-GATv2 + ACI (Ours)" in res_json["benchmark_models"]
        print("[PASS] Experimental results JSON verified.")


if __name__ == "__main__":
    test_all()

