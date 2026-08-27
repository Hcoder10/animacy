"""Motion matching: the retrieval baseline (``docs/MODEL.md`` section 5).

Index 1 s windows (30 ticks) of audio features -> the human motion that went
with them. Key = mean over the window (66) + four coarse sub-window means
(4 x 66), L2-normalised: 330-d. At run time, every 0.5 s hop picks the nearest
window (cosine), with a bonus for the window that *continues* the previous
match and for a matching speaking state, and crossfades 5 frames.

It is guaranteed to be human motion and aligned to speech; the learned model
must beat it on the held-out metrics or this ships as the default. The browser
port (``web/js/retrieval.js``) reads the index written by ``save``.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..features import N_FEATS
from .data import FEATURE_CONTRACT, MODEL_CHANNELS, N_MODEL, ClipData

WIN = 30
HOP = 15
SUBWINDOWS: List[Tuple[int, int]] = [(0, 8), (8, 15), (15, 23), (23, 30)]
KEY_DIM = N_FEATS * (1 + len(SUBWINDOWS))
CROSSFADE = 5
CONTINUITY_BONUS = 0.1
SPEAKING_BONUS = 0.05


def window_key(feats: np.ndarray) -> np.ndarray:
    """[30, 66] -> L2-normalised [330]."""
    f = np.asarray(feats, dtype=np.float32)
    parts = [f.mean(axis=0)] + [f[a:b].mean(axis=0) for a, b in SUBWINDOWS]
    k = np.concatenate(parts).astype(np.float32)
    return k / (np.linalg.norm(k) + 1e-6)


class RetrievalIndex:
    def __init__(self, keys: np.ndarray, motion: np.ndarray, next_id: np.ndarray, speaking: np.ndarray,
                 meta: Optional[Dict] = None) -> None:
        self.keys = np.asarray(keys, np.float32)          # [N, 330]
        self.motion = np.asarray(motion, np.float32)      # [N, 30, 14] raw canonical units
        self.next_id = np.asarray(next_id, np.int32)      # [N] index of the window one hop later, -1 if none
        self.speaking = np.asarray(speaking, np.float32)  # [N] fraction of speaking ticks
        self.meta = dict(meta or {})

    def __len__(self) -> int:
        return int(self.keys.shape[0])

    # ---- build --------------------------------------------------------------
    @classmethod
    def build(cls, clips: Sequence[ClipData], max_windows: int = 3000, win: int = WIN, hop: int = HOP,
              seed: int = 0) -> "RetrievalIndex":
        keys, motion, spk, nxt = [], [], [], []
        rng = np.random.default_rng(seed)
        for c in clips:
            if not c.has_audio:
                continue
            for a, b in c.runs:
                starts = list(range(a, b - win + 1, hop))
                # a clip's weight < 1 (speaker cap) keeps that fraction of its windows; the
                # continuity link only survives when the next window is kept too
                kept = [bool(k) for k in (rng.random(len(starts)) < c.weight)] if c.weight < 1.0 else [True] * len(starts)
                base = len(keys)
                pos = {}
                for si, s in enumerate(starts):
                    if not kept[si]:
                        continue
                    pos[si] = base + len(pos)
                for si, s in enumerate(starts):
                    if not kept[si]:
                        continue
                    keys.append(window_key(c.features[s:s + win]))
                    motion.append(c.motion[s:s + win])
                    spk.append(float(c.speaking[s:s + win].mean()))
                    nxt.append(pos.get(si + 1, -1))
        if not keys:
            return cls(np.zeros((0, KEY_DIM), np.float32), np.zeros((0, win, N_MODEL), np.float32),
                       np.zeros(0, np.int32), np.zeros(0, np.float32), {"win": win, "hop": hop})
        keys, motion = np.stack(keys), np.stack(motion)
        nxt, spk = np.asarray(nxt, np.int64), np.asarray(spk, np.float32)
        n = len(keys)
        if n > max_windows:
            keep = np.unique(np.round(np.linspace(0, n - 1, max_windows)).astype(np.int64))
            remap = -np.ones(n, np.int64)
            remap[keep] = np.arange(len(keep))
            nxt = np.where(nxt >= 0, remap[np.clip(nxt, 0, n - 1)], -1)[keep]
            keys, motion, spk = keys[keep], motion[keep], spk[keep]
        meta = {"win": win, "hop": hop, "n_source_windows": int(n), "n_clips": len(clips)}
        return cls(keys, motion, nxt, spk, meta)

    # ---- query --------------------------------------------------------------
    def query(self, features: np.ndarray, speaking: np.ndarray, continuity_bonus: float = CONTINUITY_BONUS,
              speaking_bonus: float = SPEAKING_BONUS, crossfade: int = CROSSFADE,
              return_ids: bool = False):
        """[T, 66], [T] -> motion [T, 14] (raw units)."""
        f = np.asarray(features, np.float32)
        s = np.asarray(speaking, np.float32)
        T = len(f)
        win, hop = int(self.meta.get("win", WIN)), int(self.meta.get("hop", HOP))
        if len(self) == 0 or T == 0:
            return np.zeros((T, N_MODEL), np.float32)
        pad = np.pad(f, ((0, win), (0, 0)), mode="edge")
        spad = np.pad(s, (0, win), mode="edge")
        out = np.zeros((T + win, N_MODEL), np.float32)
        prev = -1
        ids = []
        w = np.linspace(0.0, 1.0, crossfade + 2)[1:-1][:, None] if crossfade > 0 else None
        for h in range(0, T, hop):
            key = window_key(pad[h:h + win])
            sims = self.keys @ key
            if prev >= 0 and self.next_id[prev] >= 0:
                sims[self.next_id[prev]] += continuity_bonus
            sims += speaking_bonus * (1.0 - np.abs(self.speaking - float(spad[h:h + win].mean())))
            j = int(np.argmax(sims))
            m = self.motion[j]
            if h == 0 or crossfade <= 0:
                out[h:h + win] = m
            else:
                cf = min(crossfade, win)
                out[h:h + cf] = (1.0 - w[:cf]) * out[h:h + cf] + w[:cf] * m[:cf]
                out[h + cf:h + win] = m[cf:]
            prev = j
            ids.append(j)
        return (out[:T], np.asarray(ids)) if return_ids else out[:T]

    # ---- io -----------------------------------------------------------------
    def save(self, out_dir: str, stem: str = "retrieval") -> Tuple[str, str]:
        """``<stem>.bin`` (float16: keys then motion) + ``<stem>.json`` header."""
        os.makedirs(out_dir, exist_ok=True)
        n = len(self)
        win = int(self.meta.get("win", WIN))
        keys16 = self.keys.astype(np.float16)
        mot16 = self.motion.astype(np.float16)
        bin_path = os.path.join(out_dir, f"{stem}.bin")
        with open(bin_path, "wb") as fh:
            fh.write(keys16.tobytes())
            fh.write(mot16.tobytes())
        header = {
            "schema": "animacy.retrieval.v1",
            "feature_contract": FEATURE_CONTRACT,
            "dtype": "float16",
            "little_endian": True,
            "n": n,
            "key_dim": KEY_DIM,
            "win": win,
            "hop": int(self.meta.get("hop", HOP)),
            "rate_hz": 30,
            "channels": list(MODEL_CHANNELS),
            "subwindows": SUBWINDOWS,
            "key_recipe": "L2norm(concat(mean(f[0:30]), mean(f[a:b]) for (a,b) in subwindows))",
            "layout": {
                "keys": {"offset": 0, "shape": [n, KEY_DIM]},
                "motion": {"offset": int(keys16.nbytes), "shape": [n, win, N_MODEL]},
            },
            "next_id": self.next_id.astype(int).tolist(),
            "speaking": [round(float(x), 3) for x in self.speaking],
            "continuity_bonus": CONTINUITY_BONUS,
            "speaking_bonus": SPEAKING_BONUS,
            "crossfade": CROSSFADE,
            "bin": f"{stem}.bin",
            "meta": self.meta,
        }
        json_path = os.path.join(out_dir, f"{stem}.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(header, fh)
        return bin_path, json_path

    @classmethod
    def load(cls, json_path: str) -> "RetrievalIndex":
        header = json.load(open(json_path, encoding="utf-8"))
        bin_path = os.path.join(os.path.dirname(json_path), header["bin"])
        raw = np.fromfile(bin_path, dtype=np.float16)
        n, kd, win = header["n"], header["key_dim"], header["win"]
        keys = raw[: n * kd].reshape(n, kd).astype(np.float32)
        motion = raw[n * kd: n * kd + n * win * N_MODEL].reshape(n, win, N_MODEL).astype(np.float32)
        meta = dict(header.get("meta", {}))
        meta.update({"win": win, "hop": header["hop"]})
        return cls(keys, motion, np.asarray(header["next_id"], np.int32), np.asarray(header["speaking"], np.float32), meta)
