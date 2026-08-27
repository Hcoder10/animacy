"""Diagnostics for the motion model: is the audio actually informative?

    python scripts/model_diag.py --ckpt checkpoints/synthetic --clips checkpoints/synthetic/synthetic_clips
    python scripts/model_diag.py --ckpt checkpoints/v1 --clips data/clips

Reports, on the held-out split used by train.py:
  1. alignment: cross-correlation of the log-energy feature with mouth_open at lags -10..10 ticks
  2. a ridge probe features(15 Hz) -> standardised motion (15 Hz): held-out R^2 per channel
     (what a linear model can read off the audio at all)
  3. VQ round trip: per-channel correlation of decode(encode(gt)) with gt
  4. a2m expected motion: sum_c p(c|audio) decode(c) vs gt per channel, aligned vs block-shuffled audio
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from animacy.model.data import MODEL_CHANNELS, load_clips, normalise, pool_flag, pool_pairs, split_clips  # noqa: E402
from animacy.model.infer import MotionModel  # noqa: E402
from animacy.model.metrics import SHUFFLE_BLOCK, block_shuffle  # noqa: E402


def corr(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    clips = load_clips(a.clips, verbose=False)
    train, val, info = split_clips(clips, a.val_frac, a.seed)
    print(f"{len(clips)} clips; split {info['mode']}: train {len(train)} val {len(val)}")
    model = MotionModel.load(a.ckpt, "cpu")
    stats = model.vq.stats
    mi = MODEL_CHANNELS.index("mouth_open")

    # 1. alignment
    print("\n1. energy(feature 64) x mouth_open cross-correlation by lag (ticks at 30 Hz; + = motion lags audio)")
    lags = range(-10, 11)
    acc = {l: [] for l in lags}
    for c in val:
        for s, e in c.runs:
            en, mo = c.features[s:e, 64], c.motion[s:e, mi]
            for l in lags:
                if l >= 0:
                    acc[l].append(corr(en[: len(en) - l] if l else en, mo[l:]))
                else:
                    acc[l].append(corr(en[-l:], mo[: len(mo) + l]))
    row = " ".join(f"{l:+d}:{np.mean(acc[l]):.2f}" for l in lags)
    print("   " + row)

    # 2. ridge probe at 15 Hz
    def gather(cs):
        X, Y = [], []
        for c in cs:
            for s, e in c.runs:
                n = ((e - s) // 2) * 2
                f = pool_pairs(c.features[s:s + n])
                sp = pool_flag(c.speaking[s:s + n])[:, None].astype(np.float32)
                # +-2 step context so the probe sees onsets
                fp = np.pad(f, ((2, 2), (0, 0)), mode="edge")
                ctx = np.concatenate([fp[i:i + len(f)] for i in range(5)] + [sp], axis=1)
                X.append(ctx)
                Y.append(pool_pairs(normalise(c.motion[s:s + n], stats)))
        return np.concatenate(X), np.concatenate(Y)

    Xt, Yt = gather(train)
    Xv, Yv = gather(val)
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    Xt, Xv = (Xt - mu) / sd, (Xv - mu) / sd
    lam = 10.0
    W = np.linalg.solve(Xt.T @ Xt + lam * np.eye(Xt.shape[1]), Xt.T @ Yt)
    b = Yt.mean(0) - (Xt.mean(0) @ W)
    pred = Xv @ W + b
    r2 = 1 - ((Yv - pred) ** 2).mean(0) / (Yv.var(0) + 1e-9)
    print("\n2. ridge probe (audio +-2 steps -> standardised motion), held-out R^2 per channel:")
    print("   " + "  ".join(f"{ch}:{r2[i]:+.2f}" for i, ch in enumerate(MODEL_CHANNELS)))

    # 3. VQ round trip
    print("\n3. VQ round trip decode(encode(gt)) vs gt, held-out correlation per channel:")
    G, R = [], []
    for c in val:
        for s, e in c.runs:
            n = ((e - s) // 2) * 2
            z = normalise(c.motion[s:s + n], stats)
            R.append(model.vq.decode(model.vq.encode(z)))
            G.append(z)
    G, R = np.concatenate(G), np.concatenate(R)
    print("   " + "  ".join(f"{ch}:{corr(G[:, i], R[:, i]):+.2f}" for i, ch in enumerate(MODEL_CHANNELS)))

    # 4. a2m expected motion
    print("\n4. a2m expected decoded motion vs gt (held-out correlation per channel); aligned / block-shuffled audio")
    with torch.no_grad():
        cb = model.vq.quantizer.codebook                       # [512, dim]
        table = model.vq.decoder(cb[:, :, None]).mean(1).numpy()   # each code alone -> [512, 2, 14] -> mean pose [512, 14]
    rng = np.random.default_rng(a.seed)
    for label, shuffle in (("aligned", False), ("shuffled", True)):
        E, G = [], []
        for c in val:
            for s, e in c.runs:
                n = ((e - s) // 2) * 2
                f, sp = c.features[s:s + n], c.speaking[s:s + n]
                if shuffle:
                    f, sp = block_shuffle(f, SHUFFLE_BLOCK, rng), block_shuffle(sp, SHUFFLE_BLOCK, rng)
                lg = model.a2m.logits(pool_pairs(f), pool_flag(sp), causal=False)
                p = np.exp(lg - lg.max(1, keepdims=True))
                p /= p.sum(1, keepdims=True)
                E.append(p @ table)
                G.append(pool_pairs(normalise(c.motion[s:s + n], stats)))
        E, G = np.concatenate(E), np.concatenate(G)
        print(f"   {label:9s}" + "  ".join(f"{ch}:{corr(G[:, i], E[:, i]):+.2f}" for i, ch in enumerate(MODEL_CHANNELS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
