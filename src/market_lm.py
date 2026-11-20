"""
MarketLM: a GPT-style decoder-only transformer over market "tokens".

Tokenisation
  Every stock-day becomes one discrete token = the quantile bin of its
  volatility-normalised log return  z_t = r_t / sigma_{t-1}  (K bins fitted on training data).
Model
  Causal self-attention transformer, sequence = last CTX tokens of ONE stock,
  next-token cross-entropy (exactly the LLM objective).  A learned stock-sector-free
  'asset' embedding is not used: the model must learn from the token history itself,
  which lets it generalise to any of the ~4,000 tickers.
Outputs
  A full probability distribution over the next token -> P(up), expected move,
  and calibrated quantile range (in units of the stock's own volatility).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = Path(__file__).resolve().parent.parent
ART = BASE / "artifacts"
K = 32          # vocabulary size (return bins)
CTX = 64        # context length in trading days
VOL_WIN = 20


def tokenise(close: np.ndarray, edges: np.ndarray | None = None):
    """close: [T] prices. Returns z (vol-normalised returns) and tokens."""
    c = pd.Series(close, dtype=float)
    r = np.log(c / c.shift(1)) * 100.0
    sig = r.rolling(VOL_WIN).std().shift(1)  # causal: uses data up to t-1
    z = (r / sig.clip(lower=0.3)).clip(-6, 6)
    z = z.to_numpy()
    tok = None
    if edges is not None:
        tok = np.digitize(np.nan_to_num(z), edges)
        tok[~np.isfinite(z)] = -1
    return z, tok


class Block(nn.Module):
    def __init__(self, d, h, p=0.1):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(p))

    def forward(self, x, mask):
        a = self.ln1(x)
        x = x + self.attn(a, a, a, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class MarketLM(nn.Module):
    def __init__(self, k=K, ctx=CTX, d=96, layers=4, heads=4):
        super().__init__()
        self.k, self.ctx = k, ctx
        self.tok = nn.Embedding(k, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, k)
        self.register_buffer("mask", torch.triu(torch.ones(ctx, ctx, dtype=torch.bool), 1))

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        m = self.mask[:T, :T]
        for b in self.blocks:
            x = b(x, m)
        return self.head(self.ln(x))  # [B,T,K] next-token logits


def summarise(probs: np.ndarray, centers: np.ndarray, sigma_pct: float) -> dict:
    """Turn a next-token distribution into human-usable numbers."""
    up = float(probs[centers > 0].sum())
    ez = float((probs * centers).sum())
    cdf = np.cumsum(probs)
    q = lambda p: float(centers[min(K - 1, int(np.searchsorted(cdf, p)))])
    return {
        "prob_up": up,
        "expected_move_pct": ez * sigma_pct,
        "low_5pct": q(0.05) * sigma_pct,
        "median_pct": q(0.50) * sigma_pct,
        "high_95pct": q(0.95) * sigma_pct,
        "entropy_bits": float(-(probs * np.log2(probs + 1e-12)).sum()),
    }


_CACHE: dict = {}


def load_market_lm():
    if "m" in _CACHE:
        return _CACHE["m"]
    p = ART / "market_lm.pt"
    if not p.exists():
        return None
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = MarketLM()
    m.load_state_dict(ck["state"])
    m.eval()
    _CACHE["m"] = (m, np.array(ck["edges"]), np.array(ck["centers"]), ck["report"])
    return _CACHE["m"]


def forecast_from_closes(closes: list[float]) -> dict | None:
    bundle = load_market_lm()
    if bundle is None or len(closes) < VOL_WIN + 10:
        return None
    m, edges, centers, rep = bundle
    z, tok = tokenise(np.array(closes, dtype=float), edges)
    tok = tok[-CTX:]
    tok = tok[tok >= 0]
    if len(tok) < 8:
        return None
    r = np.diff(np.log(np.array(closes[-(VOL_WIN + 1):], dtype=float))) * 100.0
    sigma = float(max(0.3, np.std(r)))
    with torch.no_grad():
        logits = m(torch.tensor(tok, dtype=torch.long)[None])[0, -1]
        probs = F.softmax(logits, -1).numpy()
    out = summarise(probs, centers, sigma)
    out.update({"daily_vol_pct": sigma, "model_val_directional_acc": rep.get("val_directional_acc"),
                "model_val_rank_ic": rep.get("val_rank_ic"), "tokens_used": int(len(tok))})
    return out
