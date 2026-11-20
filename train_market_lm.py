"""Train MarketLM on every ticker in the 26-year file (next-token prediction)."""
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
from src.market_lm import ART, CTX, K, MarketLM, tokenise  # noqa: E402

BASE = Path(__file__).resolve().parent
torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(max(1, torch.get_num_threads()))


def main(minutes: float = 14.0) -> None:
    t0 = time.time()
    csv_matches = sorted(BASE.rglob("26YRS_closes_*.csv"))
    if not csv_matches:
        raise FileNotFoundError(f"No 26YRS_closes_*.csv dataset found under {BASE}")
    csv = csv_matches[-1]
    df = pd.read_csv(csv, parse_dates=["Date"]).set_index("Date").sort_index()
    df = df.dropna(axis=1, thresh=300).where(lambda d: d > 0)
    dates = df.index
    cut = len(dates) - 504            # last ~2y held out
    print(f"{df.shape[1]} tickers, {len(dates)} days; train<{dates[cut].date()}", flush=True)

    Z = {}
    for s in df.columns:
        z, _ = tokenise(df[s].to_numpy())
        Z[s] = z
    Zm = np.column_stack([Z[s] for s in df.columns])           # [T,N]
    # drop split/data artefacts: |z| huge days become NaN
    Zm[np.abs(Zm) >= 6.0] = np.nan
    tr = Zm[:cut]
    flat = tr[np.isfinite(tr)]
    edges = np.quantile(flat, np.linspace(0, 1, K + 1)[1:-1])
    bins_tr = np.digitize(flat, edges)
    lo_e = np.concatenate([[flat.min()], edges])
    hi_e = np.concatenate([edges, [flat.max()]])
    centers = np.array([
        np.median(flat[bins_tr == k]) if (bins_tr == k).any() else 0.5 * (lo_e[k] + hi_e[k])
        for k in range(K)
    ])
    print("centers:", np.round(centers, 2).tolist(), flush=True)
    tokens = np.where(np.isfinite(Zm), np.digitize(np.nan_to_num(Zm), edges), -1)

    def windows(lo, hi, n, rng):
        """sample n (stock, end) windows of CTX+1 valid tokens inside [lo,hi)."""
        out = []
        while len(out) < n:
            j = rng.integers(0, tokens.shape[1], n * 2)
            e = rng.integers(lo + CTX + 1, hi, n * 2)
            for jj, ee in zip(j, e):
                w = tokens[ee - CTX - 1 : ee, jj]
                if (w >= 0).all():
                    out.append(w)
                    if len(out) >= n:
                        break
        return torch.tensor(np.array(out), dtype=torch.long)

    rng = np.random.default_rng(1)
    model = MarketLM()
    print("params:", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    valset = windows(cut, len(dates), 6000, np.random.default_rng(2))

    def evaluate():
        model.eval()
        with torch.no_grad():
            lg = model(valset[:, :-1])
            nll = F.cross_entropy(lg.reshape(-1, K), valset[:, 1:].reshape(-1)).item()
            last = F.softmax(lg[:, -1], -1).numpy()
            tgt = valset[:, -1].numpy()
        up = last[:, centers > 0].sum(1)
        actual_up = centers[tgt] > 0
        acc = float(((up > 0.5) == actual_up).mean())
        base = float(max(actual_up.mean(), 1 - actual_up.mean()))
        exp = last @ centers
        ic = float(np.corrcoef(exp, centers[tgt])[0, 1])
        # uniform unigram baseline from training distribution
        pi = np.bincount(tokens[:cut][tokens[:cut] >= 0], minlength=K) / (tokens[:cut] >= 0).sum()
        base_nll = float(-np.log(pi[valset[:, 1:].numpy()] + 1e-12).mean())
        model.train()
        return nll, base_nll, acc, base, ic

    step, B = 0, 128
    model.train()
    budget = minutes * 60
    while time.time() - t0 < budget:
        frac = (time.time() - t0) / budget
        for g in opt.param_groups:
            g["lr"] = 1e-3 * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * frac)))
        x = windows(0, cut, B, rng)
        lg = model(x[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, K), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 200 == 0:
            nll, bn, acc, base, ic = evaluate()
            print(f"step {step} t={time.time()-t0:.0f}s train={loss.item():.4f} val_nll={nll:.4f} "
                  f"(unigram {bn:.4f}) dir_acc={acc:.4f} (base {base:.4f}) IC={ic:.4f}", flush=True)
    nll, bn, acc, base, ic = evaluate()
    rep = {"steps": step, "val_nll": nll, "val_unigram_nll": bn, "val_directional_acc": acc,
           "val_base_rate": base, "val_rank_ic": ic, "tickers": int(df.shape[1]),
           "val_windows": int(len(valset)), "context": CTX, "vocab": K,
           "seconds": round(time.time() - t0, 1)}
    ART.mkdir(exist_ok=True)
    torch.save({"state": model.state_dict(), "edges": edges.tolist(),
                "centers": centers.tolist(), "report": rep}, ART / "market_lm.pt")
    (BASE / "reports" / "market_lm_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2), flush=True)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 14.0)
