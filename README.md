# NSE Market Forecasting: ST-GATv2 + Adaptive Conformal Inference + MarketLM

**Cross-Industry Contagion Modeling, Distribution-Free Uncertainty Calibration, and Tokenized Market Transformer on Indian Equities (NSE/BSE)**

> **Research Project — Not Financial Advice**  
> All forecasts, confidence bands, and backtests in this repository are strictly for quantitative ML research and education. Financial time series have low signal-to-noise ratios; do not use this system for live trading.

---

## Architecture Overview

1. **Spatio-Temporal Graph Attention Network (`ST-GATv2`)** (`src/models.py`, `src/graph_builder.py`)
   - Dynamic 4-channel contagion graph across **35 liquid NSE equities** in **7 sectors** (`IT`, `Banking`, `Energy`, `Pharma`, `FMCG`, `Auto_Ind`, `Metals`) with **16 daily technical & macro features**.
   - Joint **Huber + Pinball Quantile** loss with multi-scale temporal gating and 4-head edge-conditioned graph attention, plus offline masked-feature self-supervised pre-training.
2. **Adaptive Conformal Inference (`ACI`)** (`src/conformal.py`)
   - Online distribution-free uncertainty calibration (Gibbs & Candès, 2021): $\alpha_{t+1} = \alpha_t + \gamma(\alpha^* - \text{err}_t)$, achieving **89.84%** walk-forward empirical coverage against a **90.0%** target.
3. **MarketLM — GPT-Style Market Transformer** (`src/market_lm.py`, `train_market_lm.py`)
   - Compact (~460K-parameter) causal decoder-only PyTorch transformer over 32 volatility-normalized return tokens (`context = 64` days), trained via next-token cross-entropy across 3,765 NSE/BSE series.
   - Outputs next-day rise probability, expected move, 5%–95% range, and predictive entropy via `/api/market-lm?symbol=RELIANCE`.
4. **Automated Continual Self-Supervised Learner** (`src/self_supervised.py`, `train_all_data.py`)
   - Background 3-second live quote capture + masked-feature online updates with experience replay, plus automated 24-hour Open/Close daily dataset consolidation.
5. **ConsensusSignal Backtest Pipeline** (`predictor_algo.py`)
   - Combines cross-sectional z-scores from `MarketLM` and the online learner with entropy gating, inverse-volatility market-neutral sizing, transaction costs, and a rolling hit-rate circuit breaker.

---

## Repository Structure

```text
Stock_Predictor/
├── app.py                     # FastAPI + WebSocket real-time terminal server
├── train_and_evaluate.py      # Walk-forward ST-GATv2 + ACI training, ablation & plots
├── train_market_lm.py         # PyTorch MarketLM next-token pre-training
├── train_all_data.py          # Bulk self-supervised pre-training on full multi-stock CSV
├── predictor_algo.py          # ConsensusSignal cross-sectional walk-forward backtest
├── src/
│   ├── data_pipeline.py       # Feature tensor builder (16 features, 35 equities, macro)
│   ├── graph_builder.py       # 4-channel dynamic contagion graph builder
│   ├── models.py              # ST-GATv2 + GARCH(1,1) + Ridge baselines
│   ├── conformal.py           # Split Conformal, Rolling CQR, and Adaptive Conformal Inference
│   ├── market_lm.py           # PyTorch MarketLM architecture & inference
│   ├── self_supervised.py     # Online continual self-supervised learner + replay buffer
│   ├── live_market.py         # Yahoo Finance quote/candle feed, F&O chain & analytics
│   └── metrics_and_stats.py   # Rank IC, Diebold-Mariano test, portfolio backtester
├── static/
│   └── index.html             # Interactive research terminal UI
├── artifacts/
│   ├── st_gat_aci_weights.npz # Trained ST-GATv2 weights
│   └── market_lm.pt           # Trained MarketLM checkpoint
├── reports/
│   ├── technical_report.md
│   ├── Mathematical_Model.docx
│   ├── make_model_docx.py
│   └── figures/               # Publication plots
├── tests/
│   └── test_pipeline.py       # Pipeline & mathematical property tests
├── requirements.txt
└── LICENSE
```

---

## Quickstart

### 1. Install dependencies

```powershell
# CPU installation
pip install -r requirements.txt

# Optional: GPU acceleration (NVIDIA CUDA 13.0)
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

### 2. Optional historical dataset

Large historical CSVs (e.g. `26YRS_closes_*.csv`) are excluded via `.gitignore`. Place your historical closes CSV and `India_Stock_Market_Data.xlsx` inside `data/` or `Traning Data/` — the training scripts discover them automatically.

### 3. Run the terminal server

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8050
```

Then open **http://127.0.0.1:8050** (or query **http://127.0.0.1:8050/api/market-lm?symbol=RELIANCE**).

### 4. Training & verification

```powershell
python -m tests.test_pipeline   # Verify pipeline invariants
python train_and_evaluate.py    # Train ST-GATv2 + ACI & generate figures
python train_market_lm.py 12    # Train MarketLM (minutes budget)
python train_all_data.py        # Bulk pre-train online self-supervised learner
python predictor_algo.py        # Run walk-forward ConsensusSignal backtest
```

---

## Known Limitations

- **Near-random 1-day direction:** Daily equity direction is dominated by noise (the bulk-trained online learner scores ~50.6% directional accuracy vs ~50.4% base rate; MarketLM's main gain comes from volatility clustering rather than directional timing).
- **Historical data quality:** Long-horizon close-only CSVs lack split adjustments, true OHLCV range/volume, and delisted stocks (survivorship bias).
- **Research disclaimer:** Provided under the MIT License for educational and research purposes only.
