"""Motion VQ-VAE: a codebook of movement primitives over the 14 model channels.

Ported from reachy-duplex ``training/motion_vqvae.py`` (proven on a physical
Reachy). One code covers 2 frames (15 codes/s); training windows are 8 frames.

* **Segments, not frames**: the encoder is a strided 1-D convolution over time,
  the decoder mirrors it, so a code is a *movement*, not a pose.
* **EMA codebook** updates instead of a codebook loss.
* **Dead-code revival**: codes unused for ``revive_after`` batches are re-seeded
  onto random encoder outputs. Without it a 512 codebook collapses to a handful
  of entries while the loss keeps falling (measured on reachy-duplex). Load-bearing.
* The model carries the per-channel standardisation it was trained with, so
  ``decode`` can hand back canonical units; the decoder output is bounded to
  ``+-NORM_CLIP`` standard deviations by a scaled tanh.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import FRAMES_PER_CODE, MODEL_CHANNELS, N_MODEL, NORM_CLIP


class Encoder(nn.Module):
    def __init__(self, n_channels: int, dim: int, width: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, width, 5, stride=FRAMES_PER_CODE, padding=2), nn.GELU(),
            nn.Conv1d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv1d(width, dim, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # [B, T, C] -> [B, dim, T/2]
        return self.net(x.transpose(1, 2))


class Decoder(nn.Module):
    def __init__(self, n_channels: int, dim: int, width: int = 256, out_scale: float = NORM_CLIP) -> None:
        super().__init__()
        self.out_scale = float(out_scale)
        self.net = nn.Sequential(
            nn.Conv1d(dim, width, 3, padding=1), nn.GELU(),
            nn.Conv1d(width, width, 3, padding=1), nn.GELU(),
            nn.ConvTranspose1d(width, n_channels, 4, stride=FRAMES_PER_CODE, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:      # [B, dim, L] -> [B, 2L, C]
        return (self.out_scale * torch.tanh(self.net(z) / self.out_scale)).transpose(1, 2)


class Quantizer(nn.Module):
    """Vector quantizer with EMA updates and dead-code revival."""

    def __init__(self, n_codes: int, dim: int, decay: float = 0.99, revive_after: int = 100) -> None:
        super().__init__()
        self.n_codes, self.dim, self.decay = n_codes, dim, decay
        self.revive_after = revive_after
        self.register_buffer("codebook", torch.randn(n_codes, dim) * 0.5)
        self.register_buffer("cluster_size", torch.zeros(n_codes))
        self.register_buffer("ema_embed", self.codebook.clone())
        self.register_buffer("idle", torch.zeros(n_codes))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, d, length = z.shape
        flat = z.permute(0, 2, 1).reshape(-1, d)                        # [B*L, dim]
        dist = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.codebook.t() + self.codebook.pow(2).sum(1)
        idx = dist.argmin(1)
        quant = self.codebook[idx].view(b, length, d).permute(0, 2, 1)

        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(idx, self.n_codes).type(flat.dtype)
                counts = onehot.sum(0)
                self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
                self.ema_embed.mul_(self.decay).add_(onehot.t() @ flat, alpha=1 - self.decay)
                denom = self.cluster_size.clamp(min=1e-5).unsqueeze(1)
                self.codebook.copy_(self.ema_embed / denom)
                # Revive codes that have gone unused; otherwise the codebook collapses to a
                # few entries and the model is back to a fixed menu of behaviours.
                self.idle.add_(1.0)
                self.idle[counts > 0] = 0.0
                dead = (self.idle > self.revive_after).nonzero(as_tuple=True)[0]
                if dead.numel():
                    pick = torch.randint(0, flat.shape[0], (dead.numel(),), device=flat.device)
                    self.codebook[dead] = flat[pick]
                    self.ema_embed[dead] = flat[pick]
                    self.cluster_size[dead] = 1.0
                    self.idle[dead] = 0.0

        commit = F.mse_loss(z, quant.detach())
        quant = z + (quant - z).detach()                                 # straight-through
        return quant, commit, idx.view(b, length)

    def lookup(self, idx: torch.Tensor) -> torch.Tensor:                # [B, L] -> [B, dim, L]
        return self.codebook[idx].permute(0, 2, 1)

    @torch.no_grad()
    def usage(self, idx: torch.Tensor) -> Tuple[int, float]:
        counts = torch.bincount(idx.flatten(), minlength=self.n_codes).float()
        used = int((counts > 0).sum())
        p = counts / counts.sum().clamp(min=1)
        nz = p[p > 0]
        perplexity = float(torch.exp(-(nz * nz.log()).sum()))
        return used, perplexity


class MotionVQVAE(nn.Module):
    def __init__(self, n_codes: int = 512, dim: int = 64, width: int = 256, n_channels: int = N_MODEL,
                 decay: float = 0.99, revive_after: int = 100) -> None:
        super().__init__()
        self.config = dict(n_codes=n_codes, dim=dim, width=width, n_channels=n_channels,
                           decay=decay, revive_after=revive_after)
        self.encoder = Encoder(n_channels, dim, width)
        self.quantizer = Quantizer(n_codes, dim, decay, revive_after)
        self.decoder = Decoder(n_channels, dim, width)
        self.register_buffer("mean", torch.zeros(n_channels))
        self.register_buffer("std", torch.ones(n_channels))
        self.channels = list(MODEL_CHANNELS)

    # ---- stats --------------------------------------------------------------
    def set_stats(self, stats: Dict[str, np.ndarray]) -> None:
        self.mean.copy_(torch.as_tensor(np.asarray(stats["mean"], dtype=np.float32)))
        self.std.copy_(torch.as_tensor(np.asarray(stats["std"], dtype=np.float32)))

    @property
    def stats(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean.detach().cpu().numpy().copy(), "std": self.std.detach().cpu().numpy().copy()}

    @property
    def device(self) -> torch.device:
        return self.mean.device

    @property
    def n_codes(self) -> int:
        return int(self.config["n_codes"])

    # ---- training path ------------------------------------------------------
    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        quant, commit, idx = self.quantizer(z)
        return self.decoder(quant), commit, idx

    # ---- inference ----------------------------------------------------------
    @torch.no_grad()
    def encode(self, frames_norm: np.ndarray) -> np.ndarray:
        """Standardised frames [T, C] -> codes [T//2] (an odd tail frame is dropped)."""
        was_training = self.training
        self.eval()
        x = np.asarray(frames_norm, dtype=np.float32)
        n = (len(x) // FRAMES_PER_CODE) * FRAMES_PER_CODE
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        xt = torch.from_numpy(x[:n]).to(self.device)[None]
        _, _, idx = self.quantizer(self.encoder(xt))
        if was_training:
            self.train()
        return idx[0].cpu().numpy().astype(np.int64)

    @torch.no_grad()
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Codes [L] -> standardised frames [2L, C]."""
        was_training = self.training
        self.eval()
        c = torch.as_tensor(np.asarray(codes, dtype=np.int64), device=self.device)[None]
        if c.shape[1] == 0:
            return np.zeros((0, self.config["n_channels"]), np.float32)
        out = self.decoder(self.quantizer.lookup(c))[0].cpu().numpy()
        if was_training:
            self.train()
        return out.astype(np.float32)

    def normalise(self, motion_raw: np.ndarray) -> np.ndarray:
        s = self.stats
        return np.clip((np.asarray(motion_raw, np.float32) - s["mean"]) / s["std"], -NORM_CLIP, NORM_CLIP).astype(np.float32)

    def denormalise(self, z: np.ndarray) -> np.ndarray:
        s = self.stats
        return (np.asarray(z, np.float32) * s["std"] + s["mean"]).astype(np.float32)

    # ---- io -----------------------------------------------------------------
    def save(self, path: str, extra: Optional[Dict] = None) -> str:
        torch.save({"state_dict": self.state_dict(), "config": self.config, "channels": self.channels,
                    "norm_clip": NORM_CLIP, "frames_per_code": FRAMES_PER_CODE, **(extra or {})}, path)
        return path

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "MotionVQVAE":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(**ck["config"])
        m.load_state_dict(ck["state_dict"])
        m.channels = list(ck.get("channels", MODEL_CHANNELS))
        return m.to(device).eval()
