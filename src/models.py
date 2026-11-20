"""
Spatio-Temporal Graph Attention Network (ST-GATv2) & Econometric/ML Baselines.

Implements:
  1. GARCH(1,1) + AR(1) Econometric Volatility & Return Baseline
  2. Cross-Sectional L2 Factor Regression (Ridge)
  3. Temporal-Only Gated Sequence Network (No Graph Contagion Edges)
  4. Static-Graph Convolutional Network (GCN with Fixed Sector Prior)
  5. Full Multi-Head Spatio-Temporal Graph Attention Network (ST-GATv2)
     with Joint Huber Point Forecast + Pinball Quantile Regression Heads
     (alpha/2 and 1 - alpha/2 conditional quantiles for Adaptive Conformal Inference).
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, List, Tuple

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def leaky_relu(x: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    return np.where(x > 0.0, x, alpha * x)


def dleaky_relu(x: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    return np.where(x > 0.0, 1.0, alpha).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def softmax_last_axis(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_v = np.exp(shifted)
    return exp_v / (np.sum(exp_v, axis=-1, keepdims=True) + 1e-9)


def pinball_loss_and_grad(
    y_pred_q: np.ndarray, y_true: np.ndarray, tau: float
) -> Tuple[float, np.ndarray]:
    """
    Quantile (Pinball) loss L_tau(y, q) = max(tau * (y - q), (tau - 1) * (y - q)).
    Returns scalar loss and gradient dL / dq.
    """
    err = y_true - y_pred_q
    loss = np.where(err >= 0.0, tau * err, (tau - 1.0) * err)
    grad_q = np.where(err >= 0.0, -tau, 1.0 - tau).astype(np.float32)
    grad_q /= float(y_true.size)
    return float(np.mean(loss)), grad_q


def huber_loss_and_grad(
    y_pred: np.ndarray, y_true: np.ndarray, delta: float = 1.5
) -> Tuple[float, np.ndarray]:
    """Robust Huber loss and gradient w.r.t. y_pred."""
    diff = y_pred - y_true
    abs_diff = np.abs(diff)
    quad = np.minimum(abs_diff, delta)
    lin = abs_diff - quad
    loss = 0.5 * (quad ** 2) + delta * lin
    grad = np.where(abs_diff <= delta, diff, delta * np.sign(diff)).astype(np.float32)
    grad /= float(y_true.size)
    return float(np.mean(loss)), grad


class AdamWOptimizer:
    """Decoupled Weight Decay AdamW optimizer for parameter dictionaries."""

    def __init__(
        self,
        params: Dict[str, np.ndarray],
        lr: float = 2.5e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-3,
    ) -> None:
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in params.items()}
        self.v: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads: Dict[str, np.ndarray], lr_scale: float = 1.0) -> None:
        self.t += 1
        curr_lr = self.lr * lr_scale
        b1_corr = 1.0 - (self.beta1 ** self.t)
        b2_corr = 1.0 - (self.beta2 ** self.t)

        for k, p in self.params.items():
            if k not in grads:
                continue
            g = np.clip(grads[k], -2.0, 2.0)
            if p.ndim >= 2 and self.weight_decay > 0.0:
                p -= curr_lr * self.weight_decay * p
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * (g ** 2)
            m_hat = self.m[k] / b1_corr
            v_hat = self.v[k] / b2_corr
            p -= curr_lr * m_hat / (np.sqrt(v_hat) + self.eps)


@dataclass
class STGATConfig:
    in_features: int = 14
    edge_dim: int = 4
    lookback: int = 15
    hidden_dim: int = 32
    num_heads: int = 4
    alpha_miscoverage: float = 0.10
    use_graph: bool = True
    use_dynamic_edges: bool = True
    use_tail_contagion: bool = True
    lr: float = 3e-3
    weight_decay: float = 1.5e-3
    epochs: int = 45
    seed: int = 42


class SpatioTemporalGAT:
    """
    Multi-Head Spatio-Temporal Graph Attention Network (ST-GATv2)
    with Temporal Dilated Gated Encoding + Multi-Relational Edge Attention
    + Conditional Quantile Heads.
    """

    def __init__(self, cfg: STGATConfig) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        F = cfg.in_features
        D = cfg.hidden_dim
        H = cfg.num_heads
        Dh = D // H
        De = cfg.edge_dim

        scale_in = F * 3
        self.params: Dict[str, np.ndarray] = {
            "W_temp": rng.normal(0, np.sqrt(2.0 / (scale_in + D)), (scale_in, D)).astype(np.float32),
            "b_temp": np.zeros((D,), dtype=np.float32),
            "W_gate": rng.normal(0, np.sqrt(1.0 / (scale_in + D)), (scale_in, D)).astype(np.float32),
            "b_gate": np.ones((D,), dtype=np.float32) * 0.5,
            "W_q": rng.normal(0, np.sqrt(2.0 / (D + Dh)), (H, D, Dh)).astype(np.float32),
            "W_k": rng.normal(0, np.sqrt(2.0 / (D + Dh)), (H, D, Dh)).astype(np.float32),
            "W_v": rng.normal(0, np.sqrt(2.0 / (D + Dh)), (H, D, Dh)).astype(np.float32),
            "W_e": rng.normal(0, np.sqrt(2.0 / (De + Dh)), (H, De, Dh)).astype(np.float32),
            "a_attn": rng.normal(0, np.sqrt(1.0 / Dh), (H, Dh)).astype(np.float32),
            "W_out": rng.normal(0, np.sqrt(2.0 / (D + D)), (D, D)).astype(np.float32),
            "b_out": np.zeros((D,), dtype=np.float32),
            "W_mu": rng.normal(0, 0.04, (D,)).astype(np.float32),
            "W_skip": rng.normal(0, 0.02, (F,)).astype(np.float32),
            "b_mu": np.zeros((1,), dtype=np.float32),
            "W_qlo": rng.normal(0, 0.05, (D,)).astype(np.float32),
            "b_qlo": np.array([0.85], dtype=np.float32),
            "W_qhi": rng.normal(0, 0.05, (D,)).astype(np.float32),
            "b_qhi": np.array([0.85], dtype=np.float32),
        }
        self.feat_mean: np.ndarray = np.empty(0, dtype=np.float32)
        self.feat_std: np.ndarray = np.empty(0, dtype=np.float32)

    def _build_multiscale_features(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert [T, N, F] raw features into standardized [T, N, 3*F] multi-horizon
        temporal context vectors (instantaneous t, short window [t-4:t], long window [t-L+1:t]).
        """
        if self.feat_mean.size == 0 or self.feat_std.size == 0:
            self.feat_mean = X.mean(axis=(0, 1), keepdims=True).astype(np.float32)
            self.feat_std = (X.std(axis=(0, 1), keepdims=True) + 1e-6).astype(np.float32)

        Xn = np.clip((X - self.feat_mean) / self.feat_std, -4.5, 4.5).astype(np.float32)
        T = Xn.shape[0]
        L = self.cfg.lookback

        X_short = np.zeros_like(Xn)
        X_long = np.zeros_like(Xn)
        for t in range(T):
            s5 = max(0, t - 4)
            sL = max(0, t - L + 1)
            X_short[t] = Xn[s5 : t + 1].mean(axis=0)
            X_long[t] = Xn[sL : t + 1].mean(axis=0)

        X_ms = np.concatenate([Xn, X_short, X_long], axis=-1).astype(np.float32)
        return Xn, X_ms

    def _prepare_edges(self, E_batch: np.ndarray) -> np.ndarray:
        """Apply ablation masks to edge tensor E_batch [B, N, N, De]."""
        E_mod = E_batch.copy()
        if not self.cfg.use_dynamic_edges:
            E_mod[:, :, :, 2] = 0.0
            E_mod[:, :, :, 3] = 0.0
        elif not self.cfg.use_tail_contagion:
            E_mod[:, :, :, 3] = 0.0
        return E_mod

    def forward(
        self, Xn_batch: np.ndarray, Xms_batch: np.ndarray, E_batch: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Vectorized forward pass over batch of B timestamps:
          Xn_batch:  [B, N, F]
          Xms_batch: [B, N, 3*F]
          E_batch:   [B, N, N, De]
        """
        p = self.params
        B, N, _ = Xms_batch.shape
        H = self.cfg.num_heads
        D = self.cfg.hidden_dim
        Dh = D // H

        # 1. Temporal Gated Encoding
        pre_temp = np.einsum("bnf,fd->bnd", Xms_batch, p["W_temp"], optimize=False) + p["b_temp"]
        act_temp = leaky_relu(pre_temp)
        gate_temp = sigmoid(np.einsum("bnf,fd->bnd", Xms_batch, p["W_gate"], optimize=False) + p["b_gate"])
        h_temp = act_temp * gate_temp  # [B, N, D]

        # 2. Multi-Head Spatio-Temporal Graph Attention (GATv2)
        E_used = self._prepare_edges(E_batch)
        if self.cfg.use_graph:
            W_q_flat = p["W_q"].transpose(1, 0, 2).reshape(D, H * Dh)
            W_k_flat = p["W_k"].transpose(1, 0, 2).reshape(D, H * Dh)
            W_v_flat = p["W_v"].transpose(1, 0, 2).reshape(D, H * Dh)

            q_pre = np.sum(h_temp[:, :, :, None] * W_q_flat[None, None, :, :], axis=2).reshape(B, N, H, Dh).transpose(0, 2, 1, 3)
            k_pre = np.sum(h_temp[:, :, :, None] * W_k_flat[None, None, :, :], axis=2).reshape(B, N, H, Dh).transpose(0, 2, 1, 3)
            v = np.sum(h_temp[:, :, :, None] * W_v_flat[None, None, :, :], axis=2).reshape(B, N, H, Dh).transpose(0, 2, 1, 3)

            q_act = leaky_relu(q_pre)  # [B, H, N, Dh]
            k_act = leaky_relu(k_pre)  # [B, H, N, Dh]

            q_score = np.sum(q_act * p["a_attn"][None, :, None, :], axis=-1, keepdims=True)  # [B, H, N, 1]
            k_score = np.sum(k_act * p["a_attn"][None, :, None, :], axis=-1)[:, :, None, :]  # [B, H, 1, N]

            w_edge = np.sum(p["W_e"] * p["a_attn"][:, None, :], axis=-1)  # [H, De]
            e_score = np.sum(E_used[:, None, :, :, :] * w_edge[None, :, None, None, :], axis=-1)  # [B, H, N, N]

            logits = (q_score + k_score + e_score) / np.sqrt(float(Dh))

            conn_mask = (E_used.sum(axis=-1) > 1e-5).astype(np.float32)  # [B, N, N]
            logits = logits + (1.0 - conn_mask[:, None, :, :]) * (-1e4)

            alpha = softmax_last_axis(logits)  # [B, H, N, N]
            z_heads = np.sum(alpha[:, :, :, :, None] * v[:, :, None, :, :], axis=3)
            z_concat = z_heads.transpose(0, 2, 1, 3).reshape(B, N, D)  # [B, N, D]
            g_out_pre = np.sum(z_concat[:, :, :, None] * p["W_out"][None, None, :, :], axis=2) + p["b_out"]
            g_out = leaky_relu(g_out_pre)
            u = h_temp + 0.5 * g_out
        else:
            q_pre = k_pre = q_act = k_act = v = alpha = z_concat = g_out_pre = np.zeros((1,), dtype=np.float32)
            u = h_temp

        # 3. Point Forecast & Quantile Spread Heads
        y_hat = np.sum(u * p["W_mu"][None, None, :], axis=-1) + np.sum(Xn_batch * p["W_skip"][None, None, :], axis=-1) + p["b_mu"]

        raw_lo = np.sum(u * p["W_qlo"][None, None, :], axis=-1) + p["b_qlo"]
        raw_hi = np.sum(u * p["W_qhi"][None, None, :], axis=-1) + p["b_qhi"]
        spread_lo = np.log1p(np.exp(np.clip(raw_lo, -15.0, 15.0))) + 0.15
        spread_hi = np.log1p(np.exp(np.clip(raw_hi, -15.0, 15.0))) + 0.15

        q_low = y_hat - spread_lo
        q_high = y_hat + spread_hi

        return {
            "y_hat": y_hat,
            "q_low": q_low,
            "q_high": q_high,
            "raw_lo": raw_lo,
            "raw_hi": raw_hi,
            "u": u,
            "h_temp": h_temp,
            "pre_temp": pre_temp,
            "act_temp": act_temp,
            "gate_temp": gate_temp,
            "q_pre": q_pre,
            "k_pre": k_pre,
            "q_act": q_act,
            "k_act": k_act,
            "v": v,
            "alpha": alpha,
            "z_concat": z_concat,
            "g_out_pre": g_out_pre,
            "E_used": E_used,
            "Xn": Xn_batch,
            "Xms": Xms_batch,
        }

    def backward(self, cache: Dict[str, np.ndarray], y_true: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        """Compute exact analytical gradients for joint Huber + Pinball Quantile loss."""
        p = self.params
        B, N = y_true.shape
        H = self.cfg.num_heads
        D = self.cfg.hidden_dim
        Dh = D // H

        tau_lo = self.cfg.alpha_miscoverage / 2.0
        tau_hi = 1.0 - (self.cfg.alpha_miscoverage / 2.0)

        loss_pt, grad_yhat_pt = huber_loss_and_grad(cache["y_hat"], y_true, delta=1.5)
        loss_lo, grad_qlo = pinball_loss_and_grad(cache["q_low"], y_true, tau=tau_lo)
        loss_hi, grad_qhi = pinball_loss_and_grad(cache["q_high"], y_true, tau=tau_hi)

        total_loss = loss_pt + 0.6 * (loss_lo + loss_hi)

        grad_yhat = grad_yhat_pt + 0.4 * (grad_qlo + grad_qhi)
        sig_lo = sigmoid(cache["raw_lo"])
        sig_hi = sigmoid(cache["raw_hi"])
        grad_raw_lo = 0.6 * (-grad_qlo) * sig_lo
        grad_raw_hi = 0.6 * grad_qhi * sig_hi

        grads: Dict[str, np.ndarray] = {}
        u = cache["u"]

        grads["W_mu"] = np.sum(grad_yhat[:, :, None] * u, axis=(0, 1))
        grads["W_skip"] = np.sum(grad_yhat[:, :, None] * cache["Xn"], axis=(0, 1))
        grads["b_mu"] = np.array([grad_yhat.sum()], dtype=np.float32)

        grads["W_qlo"] = np.sum(grad_raw_lo[:, :, None] * u, axis=(0, 1))
        grads["b_qlo"] = np.array([grad_raw_lo.sum()], dtype=np.float32)
        grads["W_qhi"] = np.sum(grad_raw_hi[:, :, None] * u, axis=(0, 1))
        grads["b_qhi"] = np.array([grad_raw_hi.sum()], dtype=np.float32)

        du = (
            grad_yhat[:, :, None] * p["W_mu"][None, None, :]
            + grad_raw_lo[:, :, None] * p["W_qlo"][None, None, :]
            + grad_raw_hi[:, :, None] * p["W_qhi"][None, None, :]
        )

        dh_temp = du.copy()

        if self.cfg.use_graph:
            dg_out = 0.5 * du
            dg_out_pre = dg_out * dleaky_relu(cache["g_out_pre"])
            grads["W_out"] = np.sum(cache["z_concat"][:, :, :, None] * dg_out_pre[:, :, None, :], axis=(0, 1))
            grads["b_out"] = dg_out_pre.sum(axis=(0, 1))

            dz_concat = np.sum(dg_out_pre[:, :, :, None] * p["W_out"].T[None, None, :, :], axis=2)
            dz_heads = dz_concat.reshape(B, N, H, Dh).transpose(0, 2, 1, 3)

            alpha = cache["alpha"]
            v = cache["v"]

            dv = np.sum(alpha[:, :, :, :, None] * dz_heads[:, :, :, None, :], axis=2)
            dalpha = np.sum(dz_heads[:, :, :, None, :] * v[:, :, None, :, :], axis=-1)

            sum_ad = np.sum(alpha * dalpha, axis=-1, keepdims=True)
            dlogits = alpha * (dalpha - sum_ad) / np.sqrt(float(Dh))

            dq_score = dlogits.sum(axis=3)
            dk_score = dlogits.sum(axis=2)

            q_act = cache["q_act"]
            k_act = cache["k_act"]
            dw_edge = np.sum(dlogits[:, :, :, :, None] * cache["E_used"][:, None, :, :, :], axis=(0, 2, 3))

            grads["a_attn"] = (
                np.sum(dq_score[:, :, :, None] * q_act, axis=(0, 2))
                + np.sum(dk_score[:, :, :, None] * k_act, axis=(0, 2))
                + np.sum(dw_edge[:, :, None] * p["W_e"], axis=1)
            )
            grads["W_e"] = dw_edge[:, :, None] * p["a_attn"][:, None, :]

            dq_pre = (dq_score[:, :, :, None] * p["a_attn"][None, :, None, :]) * dleaky_relu(cache["q_pre"])
            dk_pre = (dk_score[:, :, :, None] * p["a_attn"][None, :, None, :]) * dleaky_relu(cache["k_pre"])

            h_temp = cache["h_temp"]
            dq_flat = dq_pre.transpose(0, 2, 1, 3).reshape(B, N, H * Dh)
            dk_flat = dk_pre.transpose(0, 2, 1, 3).reshape(B, N, H * Dh)
            dv_flat = dv.transpose(0, 2, 1, 3).reshape(B, N, H * Dh)

            grads["W_q"] = np.sum(h_temp[:, :, :, None] * dq_flat[:, :, None, :], axis=(0, 1)).reshape(D, H, Dh).transpose(1, 0, 2)
            grads["W_k"] = np.sum(h_temp[:, :, :, None] * dk_flat[:, :, None, :], axis=(0, 1)).reshape(D, H, Dh).transpose(1, 0, 2)
            grads["W_v"] = np.sum(h_temp[:, :, :, None] * dv_flat[:, :, None, :], axis=(0, 1)).reshape(D, H, Dh).transpose(1, 0, 2)

            W_q_flat = p["W_q"].transpose(1, 0, 2).reshape(D, H * Dh)
            W_k_flat = p["W_k"].transpose(1, 0, 2).reshape(D, H * Dh)
            W_v_flat = p["W_v"].transpose(1, 0, 2).reshape(D, H * Dh)

            dh_temp += (
                np.sum(dq_flat[:, :, None, :] * W_q_flat[None, None, :, :], axis=-1)
                + np.sum(dk_flat[:, :, None, :] * W_k_flat[None, None, :, :], axis=-1)
                + np.sum(dv_flat[:, :, None, :] * W_v_flat[None, None, :, :], axis=-1)
            )

        gate_temp = cache["gate_temp"]
        act_temp = cache["act_temp"]
        dact_temp = dh_temp * gate_temp
        dgate_temp = dh_temp * act_temp

        dpre_temp = dact_temp * dleaky_relu(cache["pre_temp"])
        dpre_gate = dgate_temp * gate_temp * (1.0 - gate_temp)

        Xms = cache["Xms"]
        grads["W_temp"] = np.sum(Xms[:, :, :, None] * dpre_temp[:, :, None, :], axis=(0, 1))
        grads["b_temp"] = dpre_temp.sum(axis=(0, 1))
        grads["W_gate"] = np.sum(Xms[:, :, :, None] * dpre_gate[:, :, None, :], axis=(0, 1))
        grads["b_gate"] = dpre_gate.sum(axis=(0, 1))

        return total_loss, grads

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        E_train: np.ndarray,
        batch_size: int = 64,
    ) -> List[float]:
        """Train ST-GATv2 with mini-batch AdamW and cosine learning rate schedule."""
        Xn, Xms = self._build_multiscale_features(X_train)
        opt = AdamWOptimizer(
            self.params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        T = len(X_train)
        rng = np.random.default_rng(self.cfg.seed)
        history: List[float] = []

        for epoch in range(self.cfg.epochs):
            lr_scale = 0.5 * (1.0 + np.cos(np.pi * epoch / float(self.cfg.epochs)))
            lr_scale = max(0.15, float(lr_scale))
            perm = rng.permutation(T)
            epoch_losses: List[float] = []

            for start in range(0, T, batch_size):
                idx = perm[start : start + batch_size]
                cache = self.forward(Xn[idx], Xms[idx], E_train[idx])
                loss, grads = self.backward(cache, Y_train[idx])
                opt.step(grads, lr_scale=lr_scale)
                epoch_losses.append(loss)

            history.append(float(np.mean(epoch_losses)))
        return history

    def pretrain_self_supervised(
        self,
        X_unlabeled: np.ndarray,
        E_unlabeled: np.ndarray,
        epochs: int = 2,
        batch_size: int = 90,
        mask_ratio: float = 0.25,
    ) -> List[float]:
        """
        Self-Supervised Pre-Training (Masked Graph Feature Reconstruction):
        Randomly masks `mask_ratio` of asset features in X_unlabeled and trains the
        Temporal Encoder + GATv2 attention layers to reconstruct the masked 1-day return
        from unmasked temporal context and cross-sector graph neighbors.
        """
        Xn, Xms = self._build_multiscale_features(X_unlabeled)
        T = Xn.shape[0]
        if T < 3:
            return []
        opt = AdamWOptimizer(self.params, lr=self.cfg.lr * 0.8, weight_decay=self.cfg.weight_decay)
        rng = np.random.default_rng(self.cfg.seed + 7)
        ssl_losses: List[float] = []

        for _ in range(epochs):
            perm = rng.permutation(T - 1)
            ep_loss: List[float] = []
            for start in range(0, T - 1, batch_size):
                idx = perm[start : start + batch_size]
                mask = (rng.random(Xn[idx].shape) > mask_ratio).astype(np.float32)
                Xn_masked = Xn[idx] * mask
                Xms_masked = Xms[idx].copy()
                Xms_masked[:, :, : Xn.shape[2]] = Xn_masked
                # Self-supervised target: next-step standardized return Xn[t+1, :, 0]
                ssl_target = Xn[idx + 1, :, 0]
                cache = self.forward(Xn_masked, Xms_masked, E_unlabeled[idx])
                loss, grads = self.backward(cache, ssl_target)
                opt.step(grads, lr_scale=0.6)
                ep_loss.append(loss)
            ssl_losses.append(float(np.mean(ep_loss)))
        return ssl_losses

    def predict(
        self, X_eval: np.ndarray, E_eval: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Predict next-horizon returns, conditional quantiles, and dynamic attention maps."""
        Xn, Xms = self._build_multiscale_features(X_eval)
        out = self.forward(Xn, Xms, E_eval)
        return {
            "y_hat": out["y_hat"],
            "q_low": out["q_low"],
            "q_high": out["q_high"],
            "alpha": out["alpha"],
        }


class EconometricGARCHBaseline:
    """
    Baseline 1: AR(1) + GARCH(1,1) Econometric Volatility & Return Forecaster.
    Models conditional mean mu_{i,t} = c_i + phi_i r_{i,t} and conditional variance
    sigma^2_{i,t+1} = omega_i + alpha_i r_{i,t}^2 + beta_i sigma^2_{i,t}.
    """

    def __init__(self, alpha_miscoverage: float = 0.10) -> None:
        self.alpha_miscoverage = alpha_miscoverage
        self.c: np.ndarray = np.empty(0, dtype=np.float32)
        self.phi: np.ndarray = np.empty(0, dtype=np.float32)
        self.omega: np.ndarray = np.empty(0, dtype=np.float32)
        self.alpha_g: np.ndarray = np.empty(0, dtype=np.float32)
        self.beta_g: np.ndarray = np.empty(0, dtype=np.float32)
        self.last_var: np.ndarray = np.empty(0, dtype=np.float32)

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> None:
        r_t = X_train[:, :, 0]
        T, N = r_t.shape
        self.c = np.zeros(N, dtype=np.float32)
        self.phi = np.zeros(N, dtype=np.float32)
        self.omega = np.zeros(N, dtype=np.float32)
        self.alpha_g = np.full(N, 0.12, dtype=np.float32)
        self.beta_g = np.full(N, 0.83, dtype=np.float32)
        self.last_var = np.zeros(N, dtype=np.float32)

        for i in range(N):
            x_i = r_t[:, i]
            y_i = Y_train[:, i]
            A = np.column_stack([np.ones(T), x_i])
            coef, _, _, _ = np.linalg.lstsq(A, y_i, rcond=None)
            self.c[i] = float(coef[0])
            self.phi[i] = float(np.clip(coef[1], -0.25, 0.25))

            resid = y_i - (self.c[i] + self.phi[i] * x_i)
            uncond_var = float(np.var(resid) + 1e-4)
            self.omega[i] = uncond_var * float(1.0 - self.alpha_g[i] - self.beta_g[i])

            v = uncond_var
            for t in range(T):
                v = float(self.omega[i] + self.alpha_g[i] * (resid[t] ** 2) + self.beta_g[i] * v)
            self.last_var[i] = v

    def predict(self, X_eval: np.ndarray) -> Dict[str, np.ndarray]:
        r_t = X_eval[:, :, 0]
        T, N = r_t.shape
        y_hat = np.zeros((T, N), dtype=np.float32)
        sigma = np.zeros((T, N), dtype=np.float32)
        curr_var = self.last_var.copy()

        z_val = 1.64485 if abs(self.alpha_miscoverage - 0.10) < 0.02 else 1.95996

        for t in range(T):
            y_hat[t] = self.c + self.phi * r_t[t]
            curr_var = self.omega + self.alpha_g * (r_t[t] ** 2) + self.beta_g * curr_var
            sigma[t] = np.sqrt(np.clip(curr_var, 1e-4, 100.0))

        return {
            "y_hat": y_hat,
            "q_low": y_hat - z_val * sigma,
            "q_high": y_hat + z_val * sigma,
        }


class CrossSectionalRidgeBaseline:
    """Baseline 2: L2-Regularized Multi-Factor Cross-Sectional Linear Model."""

    def __init__(self, l2_reg: float = 50.0, alpha_miscoverage: float = 0.10) -> None:
        self.l2_reg = l2_reg
        self.alpha_miscoverage = alpha_miscoverage
        self.W: np.ndarray = np.empty(0, dtype=np.float32)
        self.b: float = 0.0
        self.mean: np.ndarray = np.empty(0, dtype=np.float32)
        self.std: np.ndarray = np.empty(0, dtype=np.float32)
        self.resid_q: float = 1.5

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> None:
        T, N, F = X_train.shape
        self.mean = X_train.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        self.std = (X_train.std(axis=(0, 1), keepdims=True) + 1e-6).astype(np.float32)
        Xn = np.clip((X_train - self.mean) / self.std, -4.0, 4.0)

        X_flat = Xn.reshape(T * N, F)
        y_flat = Y_train.reshape(T * N)
        self.b = float(y_flat.mean())
        y_c = y_flat - self.b

        A = X_flat.T @ X_flat + self.l2_reg * np.eye(F)
        self.W = np.linalg.solve(A, X_flat.T @ y_c).astype(np.float32)

        preds = X_flat @ self.W + self.b
        abs_res = np.abs(y_flat - preds)
        self.resid_q = float(np.quantile(abs_res, 1.0 - self.alpha_miscoverage))

    def predict(self, X_eval: np.ndarray) -> Dict[str, np.ndarray]:
        Xn = np.clip((X_eval - self.mean) / self.std, -4.0, 4.0)
        y_hat = np.sum(Xn * self.W[None, None, :], axis=-1) + self.b
        vol_scale = np.clip(X_eval[:, :, 3] / (X_eval[:, :, 3].mean() + 1e-6), 0.5, 2.5)
        half_w = self.resid_q * vol_scale
        return {
            "y_hat": y_hat.astype(np.float32),
            "q_low": (y_hat - half_w).astype(np.float32),
            "q_high": (y_hat + half_w).astype(np.float32),
        }
