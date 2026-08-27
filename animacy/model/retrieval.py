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
AROUSAL_BONUS = 0.15        # intent conditioning: prefer windows whose human arousal matches the target
THINKING_BONUS = 0.10       # ... and, for "thinking", windows that are still first and then move
_HEAD = [MODEL_CHANNELS.index(c) for c in ("head_yaw", "head_pitch", "head_roll")]


def _energy_var(feats: np.ndarray) -> float:
    return float(np.var(np.asarray(feats, np.float32)[:, 64]))


def _head_rms(motion: np.ndarray) -> float:
    v = np.diff(np.asarray(motion, np.float64)[:, _HEAD], axis=0) * 30.0
    return float(np.sqrt((v ** 2).sum(axis=1).mean())) if len(v) else 0.0


def _percentile_of(x, breakpoints: np.ndarray) -> np.ndarray:
    """0..1 rank of ``x`` against sorted ``breakpoints`` (the index's own distribution)."""
    bp = np.asarray(breakpoints, np.float64)
    return np.interp(np.asarray(x, np.float64), bp, np.linspace(0.0, 1.0, len(bp)))


# ---------------------------------------------------------------------------
# gesture prototypes: what a nod / head-shake / burst / tilt-and-hold / greet looks like in
# window kinematics, scored 0..1 with no labels. Exact features are documented in PROTO_DOC
# and mirrored into model.json so the browser can reproduce them.
# ---------------------------------------------------------------------------
PROTO_TAGS = ["agreement", "doubt", "excitement", "thinking", "greeting"]
PROTO_WEIGHT = 0.25
_CH = {c: MODEL_CHANNELS.index(c) for c in MODEL_CHANNELS}
_RATE = 30.0


def _unit(x: float, scale: float) -> float:
    return float(max(0.0, min(1.0, x / scale)))


def _band_ratio(x: np.ndarray, lo: float, hi: float) -> float:
    """Share of the (mean-removed) signal's power in [lo, hi] Hz."""
    x = np.asarray(x, np.float64) - np.mean(x)
    if len(x) < 8 or np.allclose(x, 0):
        return 0.0
    p = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / _RATE)
    tot = p[1:].sum()
    return float(p[(f >= lo) & (f <= hi)].sum() / tot) if tot > 0 else 0.0


def _extrema(x: np.ndarray, prominence: float, kind: str) -> int:
    from scipy.signal import find_peaks

    y = -np.asarray(x, np.float64) if kind == "min" else np.asarray(x, np.float64)
    if len(y) < 3:
        return 0
    return int(len(find_peaks(y, prominence=prominence, distance=4)[0]))


def _reversals(x: np.ndarray, min_amp: float) -> int:
    """Direction changes of a trajectory whose swings exceed ``min_amp``."""
    x = np.asarray(x, np.float64)
    if len(x) < 3:
        return 0
    n, run = 0, 0.0
    pos = x[0]
    for i in range(1, len(x)):
        d = x[i] - pos
        if abs(d) >= min_amp:
            if run != 0 and np.sign(d) != np.sign(run):
                n += 1
            run = d
            pos = x[i]
    return n


def prototype_scores(motion: np.ndarray) -> Dict[str, float]:
    """[T, 14] canonical motion (1-2 s at 30 Hz, first second = the window) -> {tag: 0..1}."""
    m = np.asarray(motion, np.float64)
    T = len(m)
    first = min(T, 30)
    pitch, yaw, roll = m[:, _CH["head_pitch"]], m[:, _CH["head_yaw"]], m[:, _CH["head_roll"]]
    z = m[:, _CH["head_z"]]
    brow = 0.5 * (m[:, _CH["brow_l"]] + m[:, _CH["brow_r"]])
    speed = np.linalg.norm(np.diff(m[:, [_CH["head_yaw"], _CH["head_pitch"], _CH["head_roll"]]], axis=0), axis=1) * _RATE
    rms = lambda x: float(np.sqrt(np.mean(np.square(np.asarray(x, np.float64) - np.mean(x))))) if len(x) else 0.0  # noqa: E731

    # agreement: pitch oscillation 1-3 Hz, >= 2 downbeats (pitch minima, prominence 1.5 deg), yaw quiet
    agreement = _unit(_band_ratio(pitch, 1.0, 3.0), 0.6) * min(1.0, _extrema(pitch, 1.5, "min") / 2.0) * float(np.exp(-rms(yaw) / 5.0))
    # doubt: yaw oscillation 0.5-3 Hz with >= 2 reversals of >= 2 deg, pitch quiet
    doubt = _unit(_band_ratio(yaw, 0.5, 3.0), 0.6) * min(1.0, _reversals(yaw, 2.0) / 2.0) * float(np.exp(-rms(pitch) / 4.0))
    # excitement: upward pitch / head_z rise in the first second with a high peak velocity, then a hold
    rise = max(float(np.max(pitch[:first]) - pitch[0]), float((np.max(z[:first]) - z[0]) / 2.0))
    vpeak = float(np.max(np.abs(np.diff(pitch[:first])) * _RATE)) if first > 1 else 0.0
    hold = 1.0 - _unit(float(np.mean(speed[first - 1:])) if T > first else 0.0, 40.0) if T > first else 1.0
    excitement = _unit(rise, 8.0) * _unit(vpeak, 60.0) * hold
    # thinking: roll/yaw tilt away (>= 5 deg), held (speed < 10 deg/s) for >= 1 s, then a small return
    exc_r, exc_y = np.abs(roll - roll[0]), np.abs(yaw - yaw[0])
    use = roll if exc_r.max() >= exc_y.max() else yaw
    exc = np.abs(use - use[0])
    k = int(np.argmax(exc))
    tilt = _unit(float(exc[k]), 8.0)
    after = speed[k:] if k < len(speed) else np.zeros(0)
    hold_frac = float(np.mean(after < 10.0)) if len(after) else 0.0
    hold_len = min(1.0, len(after) / _RATE)
    ret = 1.0 - _unit(float(exc[-1]) / max(float(exc[k]), 1e-6), 1.0) if exc[k] > 0 else 0.0
    thinking = tilt * hold_frac * hold_len * (0.5 + 0.5 * ret)
    # greeting: pitch-up onset + brow raise inside the first 0.7 s, then settle
    n07 = min(T, 21)
    pitch_up = _unit(float(np.max(pitch[:n07]) - pitch[0]), 5.0)
    brow_up = _unit(float(np.max(brow[:n07]) - brow[0]), 0.2)
    settle = 1.0 - _unit(float(np.mean(speed[n07 - 1:])) if T > n07 else 0.0, 40.0) if T > n07 else 1.0
    greeting = 0.5 * (pitch_up + brow_up) * settle
    return {"agreement": agreement, "doubt": doubt, "excitement": excitement, "thinking": thinking, "greeting": greeting}


PROTO_DOC = {
    "segment": "the 1 s window followed by its continuation window when one exists (2 s at 30 Hz); all in canonical units",
    "agreement": "band_ratio(head_pitch, 1-3 Hz)/0.6 x min(1, pitch minima with prominence >= 1.5 deg / 2) x exp(-rms(head_yaw)/5)",
    "doubt": "band_ratio(head_yaw, 0.5-3 Hz)/0.6 x min(1, yaw direction reversals of >= 2 deg / 2) x exp(-rms(head_pitch)/4)",
    "excitement": "max(pitch rise, head_z rise/2 mm) in the first second /8 deg x peak |pitch velocity| in the first second /60 deg/s x (1 - mean head speed after /40)",
    "thinking": "largest roll-or-yaw excursion from the start /8 deg x fraction of following frames with head speed < 10 deg/s x min(1, hold seconds) x (0.5 + 0.5 x return toward the start)",
    "greeting": "0.5 x (pitch rise in the first 0.7 s /5 deg + brow rise in the first 0.7 s /0.2) x (1 - mean head speed after 0.7 s /40)",
    "each factor is clipped to [0, 1]; band_ratio = share of mean-removed power in the band": True,
    "query": "score += proto_weight * proto[intent_tag][window] (proto_weight default 0.25; 0 disables)",
}


def window_key(feats: np.ndarray) -> np.ndarray:
    """[30, 66] -> L2-normalised [330]."""
    f = np.asarray(feats, dtype=np.float32)
    parts = [f.mean(axis=0)] + [f[a:b].mean(axis=0) for a, b in SUBWINDOWS]
    k = np.concatenate(parts).astype(np.float32)
    return k / (np.linalg.norm(k) + 1e-6)


class RetrievalIndex:
    def __init__(self, keys: np.ndarray, motion: np.ndarray, next_id: np.ndarray, speaking: np.ndarray,
                 meta: Optional[Dict] = None, arousal: Optional[np.ndarray] = None,
                 still_then_move: Optional[np.ndarray] = None, energy_var_breakpoints: Optional[np.ndarray] = None,
                 proto: Optional[Dict[str, np.ndarray]] = None) -> None:
        self.keys = np.asarray(keys, np.float32)          # [N, 330]
        self.motion = np.asarray(motion)                  # [N, 30, 14] raw canonical units (float32, or float16 in RAM)
        if self.motion.dtype not in (np.float16, np.float32):
            self.motion = self.motion.astype(np.float32)
        self.next_id = np.asarray(next_id, np.int32)      # [N] index of the window one hop later, -1 if none
        self.speaking = np.asarray(speaking, np.float32)  # [N] fraction of speaking ticks
        self.meta = dict(meta or {})
        n = len(self.keys)
        # intent conditioning (None on indexes built before it existed -> bonuses off)
        self.arousal = None if arousal is None else np.asarray(arousal, np.float32)                    # [N] 0..1
        self.still_then_move = None if still_then_move is None else np.asarray(still_then_move, np.float32)  # [N] -1..1
        self.energy_var_breakpoints = None if energy_var_breakpoints is None else np.asarray(energy_var_breakpoints, np.float64)
        if self.arousal is not None:
            assert len(self.arousal) == n and len(self.still_then_move) == n
        # gesture prototype scores per window, {tag: [N] 0..1}; None on older indexes
        self.proto = None if proto is None else {k: np.asarray(v, np.float32) for k, v in proto.items()}

    def compute_prototypes(self) -> None:
        """Score every window on the five gesture prototypes using the window plus its
        continuation (next_id) when it exists."""
        n = len(self)
        out = {t: np.zeros(n, np.float32) for t in PROTO_TAGS}
        for i in range(n):
            seg = self.motion[i].astype(np.float32)
            j = int(self.next_id[i])
            if j >= 0:
                seg = np.concatenate([seg, self.motion[j].astype(np.float32)], axis=0)
            s = prototype_scores(seg)
            for t in PROTO_TAGS:
                out[t][i] = s[t]
        self.proto = out

    def __len__(self) -> int:
        return int(self.keys.shape[0])

    def audio_arousal(self, feats_window: np.ndarray) -> float:
        """The audio-only proxy for a query window's arousal: its energy variance ranked
        against the index's source windows (used when no text intent is available)."""
        if self.energy_var_breakpoints is None:
            return 0.5
        return float(_percentile_of(_energy_var(feats_window), self.energy_var_breakpoints))

    @staticmethod
    def _finish(keys, motion, spk, nxt, energy_vars, head_rms, stm, meta):
        keys, motion = np.stack(keys), np.stack(motion)
        nxt, spk = np.asarray(nxt, np.int64), np.asarray(spk, np.float32)
        ev, hr, stm = np.asarray(energy_vars, np.float64), np.asarray(head_rms, np.float64), np.asarray(stm, np.float32)
        bp = np.quantile(ev, np.linspace(0, 1, 41)) if len(ev) > 1 else np.array([0.0, 1.0])
        bp = np.maximum.accumulate(bp)
        bp_h = np.maximum.accumulate(np.quantile(hr, np.linspace(0, 1, 41))) if len(hr) > 1 else np.array([0.0, 1.0])
        # a window's arousal: half how much its speech energy varies, half how much the head moved
        arousal = 0.5 * _percentile_of(ev, bp) + 0.5 * _percentile_of(hr, bp_h)
        return keys, motion, nxt, spk, arousal.astype(np.float32), stm, bp

    # ---- build --------------------------------------------------------------
    @classmethod
    def build(cls, clips: Sequence[ClipData], max_windows: int = 3000, win: int = WIN, hop: int = HOP,
              seed: int = 0) -> "RetrievalIndex":
        keys, motion, spk, nxt, ev, hr, stm = [], [], [], [], [], [], []
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
                    f, m = c.features[s:s + win], c.motion[s:s + win]
                    keys.append(window_key(f))
                    motion.append(m)
                    spk.append(float(c.speaking[s:s + win].mean()))
                    nxt.append(pos.get(si + 1, -1))
                    ev.append(_energy_var(f))
                    hr.append(_head_rms(m))
                    h1, h2 = _head_rms(m[: win // 2 + 1]), _head_rms(m[win // 2:])
                    stm.append((h2 - h1) / (h1 + h2 + 1e-6))
        if not keys:
            return cls(np.zeros((0, KEY_DIM), np.float32), np.zeros((0, win, N_MODEL), np.float32),
                       np.zeros(0, np.int32), np.zeros(0, np.float32), {"win": win, "hop": hop})
        keys, motion, nxt, spk, arousal, stm, bp = cls._finish(keys, motion, spk, nxt, ev, hr, stm, None)
        n = len(keys)
        if max_windows and n > max_windows:              # max_windows <= 0: keep every window (server-side index)
            keep = np.unique(np.round(np.linspace(0, n - 1, max_windows)).astype(np.int64))
            remap = -np.ones(n, np.int64)
            remap[keep] = np.arange(len(keep))
            nxt = np.where(nxt >= 0, remap[np.clip(nxt, 0, n - 1)], -1)[keep]
            keys, motion, spk, arousal, stm = keys[keep], motion[keep], spk[keep], arousal[keep], stm[keep]
        meta = {"win": win, "hop": hop, "n_source_windows": int(n), "n_clips": len(clips)}
        idx = cls(keys, motion, nxt, spk, meta, arousal=arousal, still_then_move=stm, energy_var_breakpoints=bp)
        idx.compute_prototypes()
        return idx

    # ---- query --------------------------------------------------------------
    def query(self, features: np.ndarray, speaking: np.ndarray, continuity_bonus: float = CONTINUITY_BONUS,
              speaking_bonus: float = SPEAKING_BONUS, crossfade: int = CROSSFADE,
              return_ids: bool = False, target_arousal: Optional[float] = None, intent_tag: Optional[str] = None,
              arousal_bonus: float = AROUSAL_BONUS, thinking_bonus: float = THINKING_BONUS,
              use_audio_arousal: bool = False, proto_weight: float = PROTO_WEIGHT):
        """[T, 66], [T] -> motion [T, 14] (raw units).

        Intent conditioning (only on indexes that carry ``arousal``): ``target_arousal`` (from
        the text) or, with ``use_audio_arousal``, the query window's own energy-variance rank
        pulls windows of matching human arousal; ``intent_tag == "thinking"`` favours
        still-then-move windows; ``proto_weight * proto[intent_tag]`` favours windows whose
        kinematics match the intent's gesture prototype (nod, head-shake, burst, tilt-hold, greet)."""
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
        use_arousal = self.arousal is not None and arousal_bonus > 0 and (target_arousal is not None or use_audio_arousal)
        proto_vec = None
        if self.proto is not None and intent_tag in self.proto and proto_weight > 0:
            proto_vec = proto_weight * self.proto[intent_tag]
        for h in range(0, T, hop):
            fw = pad[h:h + win]
            key = window_key(fw)
            sims = self.keys @ key
            if prev >= 0 and self.next_id[prev] >= 0:
                sims[self.next_id[prev]] += continuity_bonus
            sims += speaking_bonus * (1.0 - np.abs(self.speaking - float(spad[h:h + win].mean())))
            if use_arousal:
                tgt = float(target_arousal) if target_arousal is not None else self.audio_arousal(fw)
                sims += arousal_bonus * (1.0 - np.abs(self.arousal - tgt))
            if intent_tag == "thinking" and self.still_then_move is not None and thinking_bonus > 0:
                sims += thinking_bonus * np.maximum(0.0, self.still_then_move)
            if proto_vec is not None:
                sims += proto_vec
            j = int(np.argmax(sims))
            m = self.motion[j].astype(np.float32, copy=False)
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
            "ram_estimate_mb": round(self.memory_estimate(n, win)["total_mb"], 1),
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
        if self.arousal is not None:
            header.update({
                "arousal": [round(float(x), 3) for x in self.arousal],
                "still_then_move": [round(float(x), 3) for x in self.still_then_move],
                "energy_var_breakpoints": [float(x) for x in self.energy_var_breakpoints],
                "arousal_bonus": AROUSAL_BONUS,
                "thinking_bonus": THINKING_BONUS,
                "arousal_rule": "window arousal = 0.5 * rank(var of feature 64 over the window) + 0.5 * rank(head angular-speed RMS); "
                                "query bonus = arousal_bonus * (1 - |arousal - target|); target = text intent arousal, or the query "
                                "window's energy-variance rank via energy_var_breakpoints (np.interp onto 0..1) when no text is known; "
                                "thinking: + thinking_bonus * max(0, still_then_move)",
            })
        if self.proto is not None:
            header["proto"] = {t: [round(float(x), 3) for x in self.proto[t]] for t in PROTO_TAGS}
            header["proto_weight"] = PROTO_WEIGHT
            header["proto_doc"] = PROTO_DOC
        json_path = os.path.join(out_dir, f"{stem}.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(header, fh)
        return bin_path, json_path

    @staticmethod
    def memory_estimate(n: int, win: int = WIN, motion_fp16: bool = True) -> Dict[str, float]:
        """RAM per loaded index: float32 keys (the matvec operand) + motion windows."""
        keys = n * KEY_DIM * 4
        motion = n * win * N_MODEL * (2 if motion_fp16 else 4)
        return {"keys_mb": keys / 1e6, "motion_mb": motion / 1e6, "total_mb": (keys + motion) / 1e6,
                "per_window_bytes": KEY_DIM * 4 + win * N_MODEL * (2 if motion_fp16 else 4)}

    @classmethod
    def load(cls, json_path: str, motion_fp16: bool = False) -> "RetrievalIndex":
        """``motion_fp16`` keeps the motion windows as float16 in RAM (server-side indexes of
        100k+ windows: ~1.3 KB keys + 0.8 KB motion per window; a query is one [N, 330]
        float32 matvec per 0.5 s hop - brute force is fine up to a few hundred thousand windows)."""
        header = json.load(open(json_path, encoding="utf-8"))
        bin_path = os.path.join(os.path.dirname(json_path), header["bin"])
        raw = np.fromfile(bin_path, dtype=np.float16)
        n, kd, win = header["n"], header["key_dim"], header["win"]
        keys = raw[: n * kd].reshape(n, kd).astype(np.float32)
        motion = raw[n * kd: n * kd + n * win * N_MODEL].reshape(n, win, N_MODEL)
        motion = motion if motion_fp16 else motion.astype(np.float32)
        meta = dict(header.get("meta", {}))
        meta.update({"win": win, "hop": header["hop"]})
        extra = {}
        if "arousal" in header:
            extra = {"arousal": np.asarray(header["arousal"], np.float32),
                     "still_then_move": np.asarray(header["still_then_move"], np.float32),
                     "energy_var_breakpoints": np.asarray(header["energy_var_breakpoints"], np.float64)}
        if "proto" in header:
            extra["proto"] = {t: np.asarray(v, np.float32) for t, v in header["proto"].items()}
        return cls(keys, motion, np.asarray(header["next_id"], np.int32), np.asarray(header["speaking"], np.float32), meta, **extra)
