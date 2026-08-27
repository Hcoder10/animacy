"""Autoregressive audio -> codes (``a2m_ar``).

The feed-forward model in ``a2m.py`` predicts every code independently from the
audio, and speech under-determines motion, so its held-out NLL sits at the
unigram floor and its samples never settle. This model predicts
``code[t]`` from ``codes[<t]`` *and* the audio: a causal self-attention stack
over the (teacher-forced) code sequence with cross-attention into the same
audio trunk as ``AudioToMotion.encode`` (non-causal for talk mode, causal for
listen mode via the same flag; in listen mode the cross-attention is causal
too, so nothing from the future leaks through the memory).

Positions are relative everywhere (ALiBi biases + a self-attention window of
``window`` codes) so 2-4 s training chunks generalise to whole utterances, and
the ONNX graph has a dynamic length. Inference samples with temperature and
top-p; no bigram prior is needed. ``generate`` recomputes the decoder each step
over the last ``dec_layers * window`` codes with their absolute offset, which
is exactly the full recompute the browser does (asserted in the tests).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from ..features import N_FEATS
from .a2m import NEG, Attention, AudioToMotion, alibi_slopes
from .data import FEATURE_CONTRACT


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mem: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        b, T, d = x.shape
        L = mem.shape[1]
        q = self.q(x).reshape(b, T, self.h, self.dh).transpose(1, 2)              # [B, h, T, dh]
        k, v = self.kv(mem).split(d, dim=-1)
        k = k.reshape(b, L, self.h, self.dh).transpose(1, 2)
        v = v.reshape(b, L, self.h, self.dh).transpose(1, 2)
        att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dh) + bias      # [B, h, T, L]
        att = torch.softmax(att, dim=-1)
        y = torch.matmul(att, v).transpose(1, 2).reshape(b, T, d)
        return self.drop(self.out(y))


class ARBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = Attention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross = CrossAttention(d_model, n_heads, dropout)
        self.ln3 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))

    def forward(self, x, mem, self_bias, cross_bias):
        x = x + self.self_attn(self.ln1(x), self_bias)
        x = x + self.cross(self.ln2(x), mem, cross_bias)
        return x + self.mlp(self.ln3(x))


class AudioToMotionAR(nn.Module):
    def __init__(self, n_feats: int = N_FEATS, n_codes: int = 512, d_model: int = 192, enc_layers: int = 4,
                 dec_layers: int = 3, n_heads: int = 4, dropout: float = 0.25, pos_kernel: int = 7,
                 window: int = 32) -> None:
        super().__init__()
        self.config = dict(n_feats=n_feats, n_codes=n_codes, d_model=d_model, enc_layers=enc_layers,
                           dec_layers=dec_layers, n_heads=n_heads, dropout=dropout, pos_kernel=pos_kernel, window=window)
        self.n_codes, self.window, self.dec_layers = n_codes, window, dec_layers
        self.bos = n_codes
        self.encoder = AudioToMotion(n_feats, n_codes, d_model, enc_layers, n_heads, dropout, pos_kernel)
        self.code_emb = nn.Embedding(n_codes + 1, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([ARBlock(d_model, n_heads, dropout) for _ in range(dec_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_codes)
        self.register_buffer("slopes", alibi_slopes(n_heads))

    @property
    def device(self) -> torch.device:
        return self.slopes.device

    # ---- pieces ---------------------------------------------------------------
    def encode_audio(self, features, speaking, causal, key_padding_mask=None) -> torch.Tensor:
        return self.encoder.encode(features, speaking, causal, key_padding_mask)

    def decoder_biases(self, T: int, L: int, offset: int, causal: torch.Tensor, device,
                       audio_padding_mask: Optional[torch.Tensor], code_padding_mask: Optional[torch.Tensor]):
        pos_c = offset + torch.arange(T, device=device)
        pos_a = torch.arange(L, device=device)
        rel_s = pos_c[None, :] - pos_c[:, None]                                   # j - i (codes)
        self_bias = -self.slopes[:, None, None] * rel_s.abs().to(torch.float32)[None]
        blocked_s = (rel_s > 0) | (rel_s <= -self.window)                         # future, or beyond the window
        self_bias = self_bias.masked_fill(blocked_s[None], NEG)[None]             # [1, h, T, T]
        if code_padding_mask is not None:
            self_bias = self_bias.masked_fill(code_padding_mask[:, None, None, :], NEG)
        rel_x = pos_a[None, :] - pos_c[:, None]                                   # audio j - code i
        cross_bias = -self.slopes[:, None, None] * rel_x.abs().to(torch.float32)[None]
        blocked_x = (rel_x > 0) & (causal > 0)                                    # listen: no future audio
        cross_bias = cross_bias.masked_fill(blocked_x[None], NEG)[None]           # [1, h, T, L]
        if audio_padding_mask is not None:
            cross_bias = cross_bias.masked_fill(audio_padding_mask[:, None, None, :], NEG)
        return self_bias, cross_bias

    def decode(self, mem: torch.Tensor, codes_in: torch.Tensor, causal, offset: int = 0,
               audio_padding_mask=None, code_padding_mask=None) -> torch.Tensor:
        """mem [B, L, d], codes_in [B, T] (BOS-prefixed history, absolute start ``offset``)
        -> logits [B, T, n_codes]; row i predicts the code at absolute position offset + i."""
        causal = AudioToMotion.causal_tensor(causal, mem.device)
        x = self.drop(self.code_emb(codes_in))
        self_bias, cross_bias = self.decoder_biases(codes_in.shape[1], mem.shape[1], offset, causal, mem.device,
                                                    audio_padding_mask, code_padding_mask)
        for blk in self.blocks:
            x = blk(x, mem, self_bias, cross_bias)
        return self.head(self.ln_f(x))

    def forward(self, features, speaking, causal, codes_in, key_padding_mask=None, code_padding_mask=None):
        """Teacher forcing: features [B, L, 66], speaking [B, L], codes_in [B, L] = [BOS, c0 .. c_{L-2}]
        -> logits [B, L, n_codes] for c0 .. c_{L-1}."""
        mem = self.encode_audio(features, speaking, causal, key_padding_mask)
        return self.decode(mem, codes_in, causal, 0, key_padding_mask, code_padding_mask)

    # ---- inference ------------------------------------------------------------
    @staticmethod
    def sample(logits: np.ndarray, temperature: float, top_p: float, rng: np.random.Generator,
               prev: Optional[int] = None, repeat_penalty: float = 0.0) -> int:
        """Categorical draw with temperature, nucleus (top-p) and an optional penalty on
        repeating ``prev`` (motion codes repeat ~50 % of the time; a copying model freezes)."""
        z = np.asarray(logits, np.float64).copy()
        if prev is not None and 0 <= prev < len(z) and repeat_penalty:
            z[prev] -= repeat_penalty
        z = z / max(temperature, 1e-6)
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        if 0 < top_p < 1:
            order = np.argsort(-p)
            cum = np.cumsum(p[order])
            keep = order[: int(np.searchsorted(cum, top_p)) + 1]
            q = np.zeros_like(p)
            q[keep] = p[keep]
            p = q / q.sum()
        return int(rng.choice(len(p), p=p))

    @torch.no_grad()
    def generate(self, features15: np.ndarray, speaking15: np.ndarray, causal: bool = False, temperature: float = 0.8,
                 top_p: float = 0.9, seed: int = 0, repeat_penalty: float = 0.0, stay_bias: float = 0.0,
                 stay_energy: float = -0.3) -> np.ndarray:
        """[L, 66], [L] at 15 Hz -> codes [L]. Deterministic given ``seed``.
        ``stay_bias`` is the stillness knob: a logit bonus on repeating the previous code whenever
        the normalised log energy (feature 64) is below ``stay_energy`` (quiet audio)."""
        was_training = self.training
        self.eval()
        rng = np.random.default_rng(seed)
        feats = np.asarray(features15, np.float32)
        f = torch.from_numpy(feats).to(self.device)[None]
        s = torch.from_numpy(np.asarray(speaking15, np.int64)).to(self.device)[None]
        L = f.shape[1]
        energy = feats[:, 64] if feats.shape[1] > 64 else np.zeros(L, np.float32)
        mem = self.encode_audio(f, s, causal)
        ctx = self.window * self.dec_layers                 # exact: the receptive field of the windowed stack
        hist = [self.bos]
        out = []
        for t in range(L):
            tail = hist[-ctx:]
            offset = len(hist) - len(tail)
            codes_in = torch.tensor([tail], dtype=torch.int64, device=self.device)
            logits = self.decode(mem, codes_in, causal, offset)[0, -1].cpu().numpy()
            prev = out[-1] if out else None
            if prev is not None and stay_bias and energy[t] < stay_energy:
                logits = logits.copy()
                logits[prev] += stay_bias
            c = self.sample(logits, temperature, top_p, rng, prev=prev, repeat_penalty=repeat_penalty)
            out.append(c)
            hist.append(c)
        if was_training:
            self.train()
        return np.asarray(out, dtype=np.int64)

    @torch.no_grad()
    def teacher_forced_logits(self, features15: np.ndarray, speaking15: np.ndarray, codes: np.ndarray,
                              causal: bool = False) -> np.ndarray:
        """Log-likelihood scoring of a known code sequence: [L, n_codes] logits for codes[0..L-1]."""
        was_training = self.training
        self.eval()
        f = torch.from_numpy(np.asarray(features15, np.float32)).to(self.device)[None]
        s = torch.from_numpy(np.asarray(speaking15, np.int64)).to(self.device)[None]
        c = np.asarray(codes, np.int64)
        codes_in = torch.from_numpy(np.concatenate([[self.bos], c[:-1]])).to(self.device)[None]
        out = self(f, s, causal, codes_in)[0].cpu().numpy()
        if was_training:
            self.train()
        return out

    # ---- io -------------------------------------------------------------------
    def save(self, path: str, extra: Optional[Dict] = None) -> str:
        torch.save({"state_dict": self.state_dict(), "config": self.config, "arch": "ar",
                    "feature_contract": FEATURE_CONTRACT, **(extra or {})}, path)
        return path

    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(**ck["config"])
        m.load_state_dict(ck["state_dict"])
        return m.to(device).eval(), ck
