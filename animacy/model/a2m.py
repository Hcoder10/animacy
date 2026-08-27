"""Audio -> motion codes.

A small Transformer encoder (d 192, 4 layers, 4 heads, GELU) over the 15 Hz
feature stream (66-d features averaged in pairs from the 30 Hz grid) plus an
embedded ``speaking`` flag, predicting a distribution over the 512 VQ codes at
every step. One set of weights serves both modes: a ``causal`` flag switches the
attention mask (talk = non-causal, the utterance is known; listen = causal, only
the past). Training draws the flag 50/50 per batch.

Attention is written out by hand (no ``nn.MultiheadAttention``) so the ONNX
export is plain matmuls with a dynamic sequence length. Position is carried by
a causal depthwise convolution on the input and ALiBi-style distance biases, so
nothing depends on an absolute position table and the model runs on utterances
far longer than the 2 s training chunks.

``BigramPrior`` is the learned code -> code transition table used at inference
(``infer.sample_codes``) to keep sampled sequences coherent without an
autoregressive decoder in the browser.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..features import N_FEATS
from .data import FEATURE_CONTRACT

NEG = -1e4          # additive mask value (safe in fp16 too)


def alibi_slopes(n_heads: int) -> torch.Tensor:
    return torch.tensor([2.0 ** (-8.0 * (h + 1) / n_heads) for h in range(n_heads)], dtype=torch.float32)


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        b, L, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        q = q.reshape(b, L, self.h, self.dh).transpose(1, 2)           # [B, h, L, dh]
        k = k.reshape(b, L, self.h, self.dh).transpose(1, 2)
        v = v.reshape(b, L, self.h, self.dh).transpose(1, 2)
        att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dh) + bias
        att = torch.softmax(att, dim=-1)
        y = torch.matmul(att, v).transpose(1, 2).reshape(b, L, d)
        return self.drop(self.out(y))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), bias)
        return x + self.mlp(self.ln2(x))


class AudioToMotion(nn.Module):
    def __init__(self, n_feats: int = N_FEATS, n_codes: int = 512, d_model: int = 192, n_layers: int = 4,
                 n_heads: int = 4, dropout: float = 0.1, pos_kernel: int = 7) -> None:
        super().__init__()
        self.config = dict(n_feats=n_feats, n_codes=n_codes, d_model=d_model, n_layers=n_layers,
                           n_heads=n_heads, dropout=dropout, pos_kernel=pos_kernel)
        self.n_codes = n_codes
        self.pos_kernel = pos_kernel
        self.in_proj = nn.Linear(n_feats, d_model)
        self.speak_emb = nn.Embedding(2, d_model)
        self.pos_conv = nn.Conv1d(d_model, d_model, pos_kernel, groups=d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_codes)
        self.register_buffer("slopes", alibi_slopes(n_heads))

    @property
    def device(self) -> torch.device:
        return self.slopes.device

    def attention_bias(self, L: int, causal: torch.Tensor, key_padding_mask: Optional[torch.Tensor],
                       device: torch.device) -> torch.Tensor:
        """[1 or B, h, L, L]: ALiBi distance bias, future blocked when ``causal``, padding blocked."""
        pos = torch.arange(L, device=device)
        rel = pos[None, :] - pos[:, None]                               # j - i
        bias = -self.slopes[:, None, None] * rel.abs().to(torch.float32)[None]
        blocked = (rel > 0) & (causal > 0)                              # [L, L] & [1] -> [L, L]
        bias = bias.masked_fill(blocked[None], NEG)[None]               # [1, h, L, L]
        if key_padding_mask is not None:
            bias = bias.masked_fill(key_padding_mask[:, None, None, :], NEG)
        return bias

    @staticmethod
    def causal_tensor(causal, device) -> torch.Tensor:
        if not torch.is_tensor(causal):
            causal = torch.tensor([1 if causal else 0], dtype=torch.int64, device=device)
        return causal.reshape(-1)[:1]

    def encode(self, features: torch.Tensor, speaking: torch.Tensor, causal,
               key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The trunk: features [B, L, 66], speaking [B, L] -> hidden states [B, L, d_model]
        (shared with the autoregressive model in ``a2m_ar``)."""
        causal = self.causal_tensor(causal, features.device)
        x = self.in_proj(features) + self.speak_emb(speaking.clamp(0, 1))
        x = x + self.pos_conv(F.pad(x.transpose(1, 2), (self.pos_kernel - 1, 0))).transpose(1, 2)
        x = self.drop(x)
        bias = self.attention_bias(features.shape[1], causal, key_padding_mask, features.device)
        for blk in self.blocks:
            x = blk(x, bias)
        return self.ln_f(x)

    def forward(self, features: torch.Tensor, speaking: torch.Tensor, causal,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """features [B, L, 66] float, speaking [B, L] int64, causal bool | int64 tensor [1]
        -> logits [B, L, n_codes]."""
        return self.head(self.encode(features, speaking, causal, key_padding_mask))

    @torch.no_grad()
    def logits(self, features: np.ndarray, speaking: np.ndarray, causal: bool = False) -> np.ndarray:
        """Convenience for one 15 Hz sequence: [L, 66], [L] -> [L, n_codes] numpy."""
        was_training = self.training
        self.eval()
        f = torch.from_numpy(np.asarray(features, np.float32)).to(self.device)[None]
        s = torch.from_numpy(np.asarray(speaking, np.int64)).to(self.device)[None]
        out = self(f, s, causal)[0].cpu().numpy()
        if was_training:
            self.train()
        return out

    # ---- io -----------------------------------------------------------------
    def save(self, path: str, extra: Optional[Dict] = None) -> str:
        torch.save({"state_dict": self.state_dict(), "config": self.config,
                    "feature_contract": FEATURE_CONTRACT, **(extra or {})}, path)
        return path

    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(**ck["config"])
        m.load_state_dict(ck["state_dict"])
        return m.to(device).eval(), ck


class BigramPrior:
    """Code -> code transition counts with additive smoothing."""

    def __init__(self, n_codes: int, alpha: float = 0.5) -> None:
        self.n_codes, self.alpha = n_codes, alpha
        self.counts = np.zeros((n_codes, n_codes), dtype=np.float64)

    def fit(self, sequences: Sequence[np.ndarray]) -> "BigramPrior":
        for seq in sequences:
            s = np.asarray(seq, dtype=np.int64)
            if len(s) >= 2:
                np.add.at(self.counts, (s[:-1], s[1:]), 1.0)
        return self

    def log_probs(self) -> np.ndarray:
        num = self.counts + self.alpha
        return np.log(num / num.sum(axis=1, keepdims=True)).astype(np.float32)

    def unigram(self) -> np.ndarray:
        c = self.counts.sum(axis=1) + self.counts.sum(axis=0) + self.alpha
        return (c / c.sum()).astype(np.float32)
