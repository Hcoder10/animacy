"""Canonical clips → a LeRobot v3.0 dataset for one robot (``docs/LEROBOT.md``).

The writer produces the on-disk layout of ``lerobot`` 0.6.1 (``CODEBASE_VERSION
= "v3.0"``) with pyarrow/pandas only, so ``animacy`` never imports ``lerobot``.
A separate venv with ``lerobot[dataset]`` is used purely to *load and check*
what was written (:func:`validate_with_lerobot`).

Layout written under ``out_dir``::

    meta/info.json                       fps, robot_type, features, totals, splits
    meta/stats.json                      per-feature min/max/mean/std/count/q01..q99
    meta/tasks.parquet                   task string -> task_index
    meta/episodes/chunk-000/file-000.parquet   one row per episode (+ per-episode stats
                                               and animacy/* provenance columns)
    meta/animacy.json                    export provenance (clips, licenses, profile, versions)
    data/chunk-000/file-000.parquet      one row per frame, one row group per episode

One **episode** per contiguous ``face_valid`` run of at least ``min_seconds``,
long runs split into pieces of at most ``max_seconds``. The robot trajectory
comes from :func:`animacy.retarget.retarget_clip` (speed-legal, smoothed, on
the robot's rate grid); the human-side columns are re-aligned onto that grid
through the stretched timeline, so a frame's audio features, speaking flag and
canonical human channels are the ones the robot frame was computed from.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import __version__ as ANIMACY_VERSION
from .features import N_FEATS, N_MELS, SR as FEAT_SR, audio_features
from .model.data import MODEL_CHANNELS, contiguous_runs
from .profile import Profile, find_robot
from .retarget import raw_joint_targets, retarget_clip, stretch_timeline
from .schema import FACE_CHANNELS, HumanClip, MOTION_FILE

# ---- the lerobot contract this writer targets ---------------------------------
CODEBASE_VERSION = "v3.0"
LEROBOT_VERSION_TESTED = "0.6.1"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DATA_FILE_SIZE_IN_MB = 100
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"
TASKS_PATH = "meta/tasks.parquet"
PROVENANCE_PATH = "meta/animacy.json"
QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]
# lerobot's own bookkeeping columns, appended after the user features in this order.
DEFAULT_FEATURES: Dict[str, Dict] = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}

AUDIO_NAMES: List[str] = [f"mel_{i:02d}" for i in range(N_MELS)] + ["log_energy", "delta_log_energy"]
assert len(AUDIO_NAMES) == N_FEATS

OBS_STATE = "observation.state"
ACTION = "action"
OBS_HUMAN = "observation.human"
OBS_AUDIO = "observation.audio_features"
OBS_SPEAKING = "observation.speaking"
OBS_ENV_STATE = "observation.environment_state"
ENV_STATE_CHOICES = ("audio", "human", "none")


# ---------------------------------------------------------------------------
# episodes
# ---------------------------------------------------------------------------
@dataclass
class Episode:
    """One episode's frames, all on the robot grid, plus where it came from."""

    clip: str
    task: str
    state: np.ndarray          # [N, J] float32, robot units
    action: np.ndarray         # [N, J] float32, next-frame state (last frame holds)
    human: np.ndarray          # [N, 14] float32, canonical channels, NaN -> 0
    audio: np.ndarray          # [N, 66] float32, zeros when the clip has no audio
    speaking: np.ndarray       # [N] float32 0/1
    src_frames: np.ndarray     # [N] int64, source frame index in the clip's motion table
    run: Tuple[int, int]       # source frame range of the face_valid run this came from
    stretch: float             # robot frames / source frames for that run (1.0 = no time stretch)
    meta: Dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.state.shape[0])


def clean_title(title: str, max_len: int = 64) -> str:
    """``"2014-09-13 President Obama's Weekly Address.webm"`` → ``"president obama's weekly address"``."""
    s = str(title or "").strip()
    s = re.sub(r"\.(webm|ogv|mp4|mkv|mov|avi|wav|mp3)$", "", s, flags=re.I)
    s = re.sub(r"^\d{4}[-_]\d{2}[-_]\d{2}[\s_-]*", "", s)
    s = s.replace("_", " ").replace("–", " ").replace("—", " ").replace(" - ", " ").replace("- ", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    if len(s) > max_len:
        cut = s[:max_len].rsplit(" ", 1)[0]
        s = cut if cut else s[:max_len]
    return s


def task_string(meta: Dict, clip_name: str, speaking_fraction: float) -> str:
    """``"<role>: <what>"`` — role from ``meta.role`` or the VAD majority, ``what`` from
    ``meta.prompt`` / ``meta.task`` / a cleaned ``meta.title`` / the clip name."""
    role = str(meta.get("role") or ("speaking" if speaking_fraction >= 0.5 else "listening")).strip().lower()
    what = meta.get("prompt") or meta.get("task") or clean_title(meta.get("title") or "") or clean_title(clip_name)
    return f"{role}: {what}"


def _to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    if sr == FEAT_SR:
        return np.asarray(wav, dtype=np.float32)
    from scipy.signal import resample_poly

    g = math.gcd(int(sr), FEAT_SR)
    return resample_poly(np.asarray(wav, dtype=np.float64), FEAT_SR // g, int(sr) // g).astype(np.float32)


def _valid_mask(frames: pd.DataFrame) -> np.ndarray:
    valid = np.nan_to_num(frames["face_valid"].to_numpy(dtype=np.float32)) > 0
    face = frames[FACE_CHANNELS].to_numpy(dtype=np.float32)
    return valid & np.all(np.isfinite(face), axis=1)


def _split_even(n: int, max_frames: int) -> List[Tuple[int, int]]:
    pieces = max(1, int(math.ceil(n / max_frames)))
    edges = np.linspace(0, n, pieces + 1).round().astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(pieces) if edges[i + 1] > edges[i]]


def retarget_run(sub: HumanClip, profile: Profile, mode: str, fps: float):
    """Retarget one contiguous run. Returns ``(joints [M, J] float64, src_idx [M] int64)``
    on the ``fps`` grid, where ``src_idx[k]`` is the source frame whose stretched
    time is nearest to grid time ``k / fps``."""
    frames = sub.frames
    n_src = len(frames)
    raw = raw_joint_targets(frames, profile, mode)
    t_str = stretch_timeline(raw["t"].to_numpy(), raw, profile)
    t_str = t_str - t_str[0]
    table = retarget_clip(sub, profile, mode=mode)
    t_tab = table["t"].to_numpy(dtype=np.float64)
    t_last = float(t_tab[-1])
    m = int(round(t_last * fps)) + 1
    grid = np.arange(m) / fps
    joints = np.stack([np.interp(grid, t_tab, table[j].to_numpy(dtype=np.float64)) for j in profile.joint_names], axis=1)
    src = np.rint(np.interp(grid, t_str, np.arange(n_src, dtype=np.float64))).astype(np.int64)
    src = np.clip(src, 0, n_src - 1)
    return joints, src


def episodes_from_clip(clip: HumanClip, name: str, profile: Profile, mode: str = "default", fps: float = 30.0,
                       min_seconds: float = 3.0, max_seconds: float = 20.0, max_stretch: Optional[float] = 1.1,
                       dropped: Optional[List[str]] = None) -> List[Episode]:
    """Episodes of one clip. Runs whose speed-limit time stretch exceeds ``max_stretch``
    (robot frames / human frames) are dropped: their audio would no longer be real-time
    with the motion. ``dropped`` collects a line per dropped run."""
    frames = clip.frames
    n = len(frames)
    min_frames = int(round(min_seconds * clip.rate_hz))
    max_frames = max(int(round(max_seconds * fps)), int(round(min_seconds * fps)))
    runs = contiguous_runs(_valid_mask(frames), max(min_frames, 2))
    if not runs:
        return []
    human_all = np.nan_to_num(frames[MODEL_CHANNELS].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    speaking_all = (np.nan_to_num(frames["speaking"].to_numpy(dtype=np.float32)) > 0).astype(np.float32)
    has_audio = clip.audio is not None and len(clip.audio) > 0
    if has_audio:
        audio_all = audio_features(_to_16k(clip.audio, clip.sr), FEAT_SR, n_ticks=n, rate_hz=clip.rate_hz).astype(np.float32)
    else:
        audio_all = np.zeros((n, N_FEATS), dtype=np.float32)
    out: List[Episode] = []
    for a, b in runs:
        sub_frames = frames.iloc[a:b].reset_index(drop=True).copy()
        sub_frames["t"] = sub_frames["t"] - float(sub_frames["t"].iloc[0])
        sub = HumanClip.from_frames(sub_frames, **clip.meta)
        joints, src_local = retarget_run(sub, profile, mode, fps)
        m = len(joints)
        if m < int(round(min_seconds * fps)):
            continue
        stretch = m / float(b - a)
        if max_stretch is not None and stretch > max_stretch:
            if dropped is not None:
                dropped.append(f"{name}: run {a}:{b} ({(b - a) / clip.rate_hz:.1f}s) time-stretched {stretch:.3f}x > {max_stretch}")
            continue
        state = joints.astype(np.float32)
        action = np.concatenate([state[1:], state[-1:]], axis=0)
        src = src_local + a
        for s, e in _split_even(m, max_frames):
            sp = speaking_all[src[s:e]]
            out.append(Episode(
                clip=name,
                task=task_string(clip.meta, name, float(sp.mean()) if len(sp) else 0.0),
                state=state[s:e], action=action[s:e],
                human=human_all[src[s:e]], audio=audio_all[src[s:e]], speaking=sp,
                src_frames=src[s:e], run=(a, b), stretch=stretch,
                meta={"has_audio": has_audio, "subject": str(clip.meta.get("subject") or ""),
                      "source": str(clip.meta.get("source") or ""), "source_url": clip.meta.get("source_url"),
                      "license": clip.meta.get("license"), "title": clip.meta.get("title")},
            ))
    return out


# ---------------------------------------------------------------------------
# features / stats
# ---------------------------------------------------------------------------
def build_features(profile: Profile, env_state: str = "audio") -> Dict[str, Dict]:
    if env_state not in ENV_STATE_CHOICES:
        raise ValueError(f"env_state must be one of {ENV_STATE_CHOICES}, got {env_state!r}")
    j = profile.joint_names
    feats: Dict[str, Dict] = {
        OBS_STATE: {"dtype": "float32", "shape": (len(j),), "names": list(j)},
        ACTION: {"dtype": "float32", "shape": (len(j),), "names": list(j)},
        OBS_HUMAN: {"dtype": "float32", "shape": (len(MODEL_CHANNELS),), "names": list(MODEL_CHANNELS)},
        OBS_AUDIO: {"dtype": "float32", "shape": (N_FEATS,), "names": list(AUDIO_NAMES)},
        OBS_SPEAKING: {"dtype": "float32", "shape": (1,), "names": None},
    }
    if env_state == "audio":
        feats[OBS_ENV_STATE] = {"dtype": "float32", "shape": (N_FEATS + 1,), "names": list(AUDIO_NAMES) + ["speaking"]}
    elif env_state == "human":
        feats[OBS_ENV_STATE] = {"dtype": "float32", "shape": (len(MODEL_CHANNELS) + 1,), "names": list(MODEL_CHANNELS) + ["speaking"]}
    feats.update({k: dict(v) for k, v in DEFAULT_FEATURES.items()})
    return feats


def episode_columns(ep: Episode, env_state: str) -> Dict[str, np.ndarray]:
    """The user-feature columns of one episode as 2-D arrays ``[N, d]``."""
    sp = ep.speaking.reshape(-1, 1).astype(np.float32)
    cols = {OBS_STATE: ep.state, ACTION: ep.action, OBS_HUMAN: ep.human, OBS_AUDIO: ep.audio, OBS_SPEAKING: sp}
    if env_state == "audio":
        cols[OBS_ENV_STATE] = np.concatenate([ep.audio, sp], axis=1).astype(np.float32)
    elif env_state == "human":
        cols[OBS_ENV_STATE] = np.concatenate([ep.human, sp], axis=1).astype(np.float32)
    return cols


def feature_stats(x: np.ndarray) -> Dict[str, np.ndarray]:
    """lerobot-shaped stats for a ``[N, d]`` column: min/max/mean/std (population),
    exact quantiles, and ``count = [N]``. Every value is a 1-D array of length ``d``."""
    x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    st = {
        "min": x.min(axis=0), "max": x.max(axis=0), "mean": x.mean(axis=0), "std": x.std(axis=0),
        "count": np.array([x.shape[0]], dtype=np.int64),
    }
    qs = np.quantile(x, QUANTILES, axis=0)
    for q, row in zip(QUANTILES, qs):
        st[f"q{int(q * 100):02d}"] = np.asarray(row, dtype=np.float64).reshape(-1)
    return st


def _serialize_stats(stats: Dict[str, Dict[str, np.ndarray]]) -> Dict:
    return {k: {s: (v.astype(int).tolist() if s == "count" else np.asarray(v, dtype=np.float64).tolist())
                for s, v in d.items()} for k, d in stats.items()}


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------
def _arrow_column(x: np.ndarray, dtype: str, shape: Tuple[int, ...]):
    import pyarrow as pa

    pa_type = {"float32": pa.float32(), "float64": pa.float64(), "int64": pa.int64(), "int32": pa.int32(), "bool": pa.bool_()}[dtype]
    if tuple(shape) == (1,):
        return pa.array(np.asarray(x).reshape(-1).astype(dtype), type=pa_type)
    d = int(shape[0])
    flat = pa.array(np.ascontiguousarray(np.asarray(x).reshape(-1, d).astype(dtype)).reshape(-1), type=pa_type)
    return pa.FixedSizeListArray.from_arrays(flat, d)


def _next_file(chunk_idx: int, file_idx: int, chunks_size: int) -> Tuple[int, int]:
    file_idx += 1
    if file_idx >= chunks_size:
        return chunk_idx + 1, 0
    return chunk_idx, file_idx


def _estimate_bytes_per_frame(features: Dict[str, Dict]) -> int:
    n = 0
    for ft in features.values():
        n += int(np.prod(ft["shape"])) * np.dtype(ft["dtype"]).itemsize
    return n


def write_dataset(episodes: Sequence[Episode], profile: Profile, out_dir: str, fps: float = 30.0,
                  env_state: str = "audio", provenance: Optional[Dict] = None,
                  data_files_size_in_mb: int = DEFAULT_DATA_FILE_SIZE_IN_MB,
                  chunks_size: int = DEFAULT_CHUNK_SIZE) -> Dict:
    """Write the LeRobot v3.0 tree. Returns the ``info.json`` dict plus a few summary keys."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not episodes:
        raise ValueError("no episodes to write")
    if abs(fps - round(fps)) > 1e-9:
        raise ValueError(f"lerobot fps must be an integer, got {fps}")
    fps_i = int(round(fps))
    features = build_features(profile, env_state)
    os.makedirs(os.path.join(out_dir, "meta"), exist_ok=True)

    # tasks: first-seen order, like lerobot's save_episode_tasks
    tasks: List[str] = []
    for ep in episodes:
        if ep.task not in tasks:
            tasks.append(ep.task)
    task_index = {t: i for i, t in enumerate(tasks)}

    # ---- data/ ---------------------------------------------------------------
    bytes_per_frame = _estimate_bytes_per_frame(features)
    max_frames_per_file = max(1, int(data_files_size_in_mb * 1024 * 1024 / max(bytes_per_frame, 1)))
    chunk_idx, file_idx = 0, 0
    writer = None
    frames_in_file = 0
    global_index = 0
    ep_rows: List[Dict] = []
    all_cols: Dict[str, List[np.ndarray]] = {k: [] for k in features}
    for ep_i, ep in enumerate(episodes):
        n = len(ep)
        if writer is not None and frames_in_file + n > max_frames_per_file:
            writer.close()
            writer = None
            chunk_idx, file_idx = _next_file(chunk_idx, file_idx, chunks_size)
            frames_in_file = 0
        cols = episode_columns(ep, env_state)
        cols["timestamp"] = (np.arange(n, dtype=np.float64) / fps_i).astype(np.float32).reshape(-1, 1)
        cols["frame_index"] = np.arange(n, dtype=np.int64).reshape(-1, 1)
        cols["episode_index"] = np.full((n, 1), ep_i, dtype=np.int64)
        cols["index"] = np.arange(global_index, global_index + n, dtype=np.int64).reshape(-1, 1)
        cols["task_index"] = np.full((n, 1), task_index[ep.task], dtype=np.int64)
        arrays = [_arrow_column(cols[k], features[k]["dtype"], features[k]["shape"]) for k in features]
        table = pa.Table.from_arrays(arrays, names=list(features))
        if writer is None:
            path = os.path.join(out_dir, DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            writer = pq.ParquetWriter(path, table.schema, compression="snappy", use_dictionary=True)
        writer.write_table(table)  # one row group per episode, like lerobot's writer
        frames_in_file += n
        ep_stats = {k: feature_stats(cols[k]) for k in features}
        row: Dict = {
            "episode_index": ep_i, "tasks": [ep.task], "length": n,
            "data/chunk_index": chunk_idx, "data/file_index": file_idx,
            "dataset_from_index": global_index, "dataset_to_index": global_index + n,
            "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0,
            "animacy/clip": ep.clip, "animacy/run_start_frame": int(ep.run[0]), "animacy/run_end_frame": int(ep.run[1]),
            "animacy/src_frame_start": int(ep.src_frames[0]), "animacy/src_frame_end": int(ep.src_frames[-1]) + 1,
            "animacy/stretch": float(ep.stretch), "animacy/speaking_fraction": float(ep.speaking.mean()),
            "animacy/has_audio": bool(ep.meta.get("has_audio", False)),
            "animacy/subject": str(ep.meta.get("subject") or ""), "animacy/license": str(ep.meta.get("license") or ""),
            "animacy/source_url": str(ep.meta.get("source_url") or ""),
        }
        for k, st in ep_stats.items():
            for s, v in st.items():
                row[f"stats/{k}/{s}"] = v.astype(int).tolist() if s == "count" else np.asarray(v, dtype=np.float64).tolist()
        ep_rows.append(row)
        for k in features:
            all_cols[k].append(cols[k])
        global_index += n
    if writer is not None:
        writer.close()
    total_frames = global_index

    # ---- meta/episodes ---------------------------------------------------------
    ep_path = os.path.join(out_dir, EPISODES_PATH.format(chunk_index=0, file_index=0))
    os.makedirs(os.path.dirname(ep_path), exist_ok=True)
    ep_table = pa.Table.from_pydict({k: [r[k] for r in ep_rows] for k in ep_rows[0]})
    pq.write_table(ep_table, ep_path, compression="snappy", use_dictionary=True)

    # ---- meta/tasks ------------------------------------------------------------
    tasks_df = pd.DataFrame({"task_index": list(range(len(tasks)))}, index=pd.Index(tasks, name="task"))
    tasks_df.to_parquet(os.path.join(out_dir, TASKS_PATH))

    # ---- meta/stats ------------------------------------------------------------
    stats = {k: feature_stats(np.concatenate(all_cols[k], axis=0)) for k in features}
    with open(os.path.join(out_dir, STATS_PATH), "w", encoding="utf-8") as fh:
        json.dump(_serialize_stats(stats), fh, indent=4, ensure_ascii=False)

    # ---- meta/info -------------------------------------------------------------
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps_i,
        "features": {k: {"dtype": ft["dtype"], "shape": list(ft["shape"]), "names": ft["names"]} for k, ft in features.items()},
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "chunks_size": chunks_size,
        "data_files_size_in_mb": data_files_size_in_mb,
        "video_files_size_in_mb": DEFAULT_VIDEO_FILE_SIZE_IN_MB,
        "data_path": DATA_PATH,
        "video_path": None,
        "robot_type": f"animacy/{profile.name}",
        "splits": {"train": f"0:{len(episodes)}"},
    }
    with open(os.path.join(out_dir, INFO_PATH), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=4, ensure_ascii=False)

    # ---- meta/animacy.json (provenance; lerobot ignores it) --------------------
    clips: Dict[str, Dict] = {}
    for ep in episodes:
        c = clips.setdefault(ep.clip, {"episodes": 0, "frames": 0, "tasks": [], **{k: ep.meta.get(k) for k in ("subject", "source", "source_url", "license", "title", "has_audio")}})
        c["episodes"] += 1
        c["frames"] += len(ep)
        if ep.task not in c["tasks"]:
            c["tasks"].append(ep.task)
    prov = {
        "schema": "animacy.lerobot_export.v1",
        "animacy_version": ANIMACY_VERSION,
        "lerobot_codebase_version": CODEBASE_VERSION,
        "lerobot_version_tested": LEROBOT_VERSION_TESTED,
        "robot": {"name": profile.name, "display_name": profile.display_name, "path": os.path.relpath(profile.path, os.getcwd()) if profile.path else "",
                  "rate_hz": profile.rate_hz, "joints": [j.model_dump() for j in profile.joints]},
        "fps": fps_i,
        "env_state": env_state,
        "action_definition": "action[t] = observation.state[t+1] within the same face_valid run; the run's last frame holds",
        "alignment": "human/audio/speaking columns are indexed by the source frame nearest in stretched time (see animacy/src_frame_* per episode)",
        "clips": clips,
        "tasks": tasks,
        **(provenance or {}),
    }
    with open(os.path.join(out_dir, PROVENANCE_PATH), "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2, ensure_ascii=False, default=str)
    return {**info, "tasks": tasks, "clips": clips, "out_dir": out_dir}


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------
def list_clip_dirs(clips_dir: str, exclude: Sequence[str] = ()) -> List[str]:
    if os.path.exists(os.path.join(clips_dir, MOTION_FILE)):
        return [clips_dir]
    out = []
    for d in sorted(os.listdir(clips_dir)):
        p = os.path.join(clips_dir, d)
        if d in exclude or not os.path.isdir(p) or not os.path.exists(os.path.join(p, MOTION_FILE)):
            continue
        out.append(p)
    return out


def _prepare_out_dir(out_dir: str, force: bool) -> None:
    if not os.path.exists(out_dir) or not os.listdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        return
    if not force:
        raise FileExistsError(f"{out_dir} exists and is not empty (use --force to replace it)")
    if not os.path.exists(os.path.join(out_dir, INFO_PATH)):
        raise FileExistsError(f"refusing to --force-remove {out_dir}: it does not look like a LeRobot dataset (no {INFO_PATH})")
    shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)


def export(robot: str, clips_dir: str, out_dir: str, fps: float = 30.0, mode: str = "default",
           exclude: Sequence[str] = (), min_seconds: float = 3.0, max_seconds: float = 20.0,
           env_state: str = "audio", max_stretch: Optional[float] = 1.1, force: bool = False,
           verbose: bool = True) -> Dict:
    profile = find_robot(robot)
    errs = profile.check()
    errs = [e for e in errs if "urdf" not in e.lower() and "native_clips" not in e]
    if errs:
        raise ValueError(f"robot {profile.name} fails `animacy check`: {errs}")
    if mode not in profile.retarget:
        raise KeyError(f"robot {profile.name} has no retarget mode {mode!r}; modes: {list(profile.retarget)}")
    _prepare_out_dir(out_dir, force)
    episodes: List[Episode] = []
    skipped: List[str] = []
    for p in list_clip_dirs(clips_dir, exclude):
        name = os.path.basename(p)
        try:
            clip = HumanClip.load(p, audio=True)
        except Exception as e:  # half-written clip must not kill the export
            skipped.append(f"{name}: {type(e).__name__}: {e}")
            continue
        probs = clip.validate()
        eps = episodes_from_clip(clip, name, profile, mode=mode, fps=fps, min_seconds=min_seconds, max_seconds=max_seconds,
                                 max_stretch=max_stretch, dropped=skipped)
        if verbose:
            n_fr = sum(len(e) for e in eps)
            print(f"  {name}: {len(clip)} frames -> {len(eps)} episodes, {n_fr} frames ({n_fr / fps:.1f}s), "
                  f"audio={'yes' if clip.audio is not None else 'NO'}"
                  + (f", clip problems: {probs}" if probs else ""))
        if not eps:
            skipped.append(f"{name}: no face_valid run >= {min_seconds}s")
        episodes.extend(eps)
    if not episodes:
        raise ValueError(f"no episodes from {clips_dir} (skipped: {skipped})")
    prov = {"clips_dir": os.path.abspath(clips_dir), "mode": mode, "min_seconds": min_seconds, "max_seconds": max_seconds,
            "max_stretch": max_stretch, "excluded": list(exclude), "skipped": skipped}
    summary = write_dataset(episodes, profile, out_dir, fps=fps, env_state=env_state, provenance=prov)
    summary["skipped"] = skipped
    summary["stretch_max"] = max(e.stretch for e in episodes)
    return summary


# ---------------------------------------------------------------------------
# validation in the lerobot venv
# ---------------------------------------------------------------------------
def default_lerobot_python(repo_root: Optional[str] = None) -> Optional[str]:
    """``$ANIMACY_LEROBOT_PYTHON`` or ``<repo>/.venv-lerobot``'s interpreter, if present."""
    env = os.environ.get("ANIMACY_LEROBOT_PYTHON")
    if env and os.path.exists(env):
        return env
    root = repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for cand in (os.path.join(root, ".venv-lerobot", "Scripts", "python.exe"), os.path.join(root, ".venv-lerobot", "bin", "python")):
        if os.path.exists(cand):
            return cand
    return None


# Runs inside the lerobot venv. Prints one JSON line prefixed with VALIDATION_JSON on success.
VALIDATOR_SRC = r'''
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
root, repo_id = sys.argv[1], sys.argv[2]
import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset, CODEBASE_VERSION
from lerobot.datasets.feature_utils import get_hf_features_from_features
import pyarrow.parquet as pq
from pathlib import Path

ds = LeRobotDataset(repo_id, root=root)
info = ds.meta.info
assert info.codebase_version == CODEBASE_VERSION, (info.codebase_version, CODEBASE_VERSION)
feats = ds.meta.features
n = len(ds)
assert n == info.total_frames == ds.meta.total_frames, (n, info.total_frames)
assert ds.num_episodes == info.total_episodes
assert ds.meta.tasks is not None and len(ds.meta.tasks) == info.total_tasks
# parquet schema == what lerobot would have written for these features
expected = get_hf_features_from_features(feats).arrow_schema
data_files = sorted((Path(root) / "data").glob("*/*.parquet"))
for f in data_files:
    got = pq.read_schema(str(f))
    for name in expected.names:
        assert got.field(name).type == expected.field(name).type, (f.name, name, got.field(name).type, expected.field(name).type)
# items
fps = info.fps
checked = 0
for i in sorted({0, n // 3, n // 2, n - 1}):
    item = ds[i]
    for k, ft in feats.items():
        v = item[k]
        assert isinstance(v, torch.Tensor), (k, type(v))
        shape = tuple(ft["shape"]) if tuple(ft["shape"]) != (1,) else ()
        assert tuple(v.shape) == shape, (k, tuple(v.shape), shape)
        want = {"float32": torch.float32, "int64": torch.int64}[ft["dtype"]]
        assert v.dtype == want, (k, v.dtype, want)
        assert torch.isfinite(v.float()).all(), (k, "non-finite")
    assert isinstance(item["task"], str) and item["task"]
    assert abs(float(item["timestamp"]) - int(item["frame_index"]) / fps) < 1e-4
    checked += 1
# episode boundaries + action = next state
ep0 = ds.meta.episodes[0]
a, b = int(ep0["dataset_from_index"]), int(ep0["dataset_to_index"])
assert b - a == int(ep0["length"]) and b - a >= 3 * fps
st = torch.stack([ds[i]["observation.state"] for i in range(a, min(b, a + 5))])
ac = torch.stack([ds[i]["action"] for i in range(a, min(b, a + 5))])
assert torch.allclose(ac[:-1], st[1:]), "action[t] != state[t+1]"
# stats
for k, ft in feats.items():
    s = ds.meta.stats[k]
    d = int(np.prod(ft["shape"]))
    for key in ("min", "max", "mean", "std", "q01", "q50", "q99"):
        assert s[key].shape == (d,), (k, key, s[key].shape)
    assert s["count"].shape == (1,) and int(s["count"][0]) == n, (k, s["count"])
# delta timestamps (action chunking like ACT) + a default-collate batch
chunk = 10
ds2 = LeRobotDataset(repo_id, root=root, delta_timestamps={"action": [i / fps for i in range(chunk)]})
it = ds2[a]
assert tuple(it["action"].shape) == (chunk, feats["action"]["shape"][0]), it["action"].shape
assert it["action_is_pad"].shape == (chunk,) and not bool(it["action_is_pad"].any())
loader = torch.utils.data.DataLoader(ds2, batch_size=8, shuffle=True, num_workers=0)
batch = next(iter(loader))
assert tuple(batch["observation.state"].shape) == (8, feats["observation.state"]["shape"][0])
assert tuple(batch["action"].shape) == (8, chunk, feats["action"]["shape"][0])
assert len(batch["task"]) == 8
# policy-side feature typing (what lerobot-train would build)
from lerobot.utils.feature_utils import dataset_to_policy_features
pf = dataset_to_policy_features(feats)
out = {
    "lerobot_version": lerobot.__version__, "codebase_version": info.codebase_version,
    "robot_type": info.robot_type, "fps": fps, "total_episodes": info.total_episodes, "total_frames": info.total_frames,
    "total_tasks": info.total_tasks, "tasks": list(ds.meta.tasks.index), "splits": info.splits,
    "features": {k: {"dtype": ft["dtype"], "shape": list(ft["shape"])} for k, ft in feats.items()},
    "policy_features": {k: str(v.type.value) for k, v in pf.items()},
    "items_checked": checked, "data_files": [f.name for f in data_files],
    "episode_lengths_s": {"min": round(min(ds.meta.episodes["length"]) / fps, 2), "max": round(max(ds.meta.episodes["length"]) / fps, 2)},
    "batch_ok": True,
}
print("VALIDATION_JSON " + json.dumps(out))
'''


def validate_with_lerobot(out_dir: str, repo_id: str = "animacy/local", python_exe: Optional[str] = None,
                          timeout: int = 900) -> Tuple[bool, str, Optional[Dict]]:
    """Load ``out_dir`` with the real ``LeRobotDataset`` in the lerobot venv and check it.
    Returns ``(ok, log, summary)``; ``ok`` is False (with an explanation) when no venv is found."""
    py = python_exe or default_lerobot_python()
    if not py:
        return False, "no lerobot venv found (set ANIMACY_LEROBOT_PYTHON or create .venv-lerobot; see docs/LEROBOT.md)", None
    # -I: isolated mode. A PYTHONPATH pointing at another venv (this machine exports one for
    # reachy-duplex) would otherwise shadow the lerobot venv's own packages (seen: av 18 vs 15).
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    env.update({"PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1"})
    try:
        proc = subprocess.run([py, "-I", "-X", "utf8", "-W", "ignore", "-c", VALIDATOR_SRC, os.path.abspath(out_dir), repo_id],
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return False, f"validator timed out after {timeout}s", None
    log = (proc.stdout or "") + (proc.stderr or "")
    summary = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("VALIDATION_JSON "):
            summary = json.loads(line[len("VALIDATION_JSON "):])
    ok = proc.returncode == 0 and summary is not None
    return ok, log, summary


# ---------------------------------------------------------------------------
# hub
# ---------------------------------------------------------------------------
def dataset_card(info: Dict, prov: Dict, repo_id: str, license_id: str = "cc-by-4.0") -> str:
    robot = info.get("robot_type", "")
    tags = ["LeRobot", "animacy", "expressive-robots", "vla", robot.split("/")[-1].replace("_", "-")]
    lines = [
        "---",
        f"license: {license_id}",
        "task_categories:",
        "- robotics",
        "tags:",
        *[f"- {t}" for t in tags],
        "configs:",
        "- config_name: default",
        "  data_files: data/*/*.parquet",
        "---",
        "",
        f"# {repo_id}",
        "",
        f"Human conversational motion retargeted to **{robot}** as a LeRobot v3.0 dataset, produced by",
        "[animacy](https://github.com/Hcoder10/animacy) (`scripts/export_lerobot.py`). The robot joint",
        "vector is `observation.state`, the next frame is `action`, and every frame carries the",
        "speech features, the speaking flag and the canonical human channels it was computed from,",
        "so the same dataset trains robot-space and human-space policies.",
        "",
        f"**{info['total_episodes']} episodes, {info['total_frames']} frames at {info['fps']} fps "
        f"({info['total_frames'] / info['fps'] / 60:.1f} min), {info['total_tasks']} task strings.**",
        "",
        "```python",
        "from lerobot.datasets.lerobot_dataset import LeRobotDataset",
        f'ds = LeRobotDataset("{repo_id}")',
        "item = ds[0]   # observation.state, action, observation.human, observation.audio_features, observation.speaking, ...",
        "```",
        "",
        "## Features",
        "",
        "| key | dtype | shape | names |",
        "|---|---|---|---|",
    ]
    for k, ft in info["features"].items():
        names = ft.get("names")
        ns = ", ".join(names[:8]) + (", ..." if names and len(names) > 8 else "") if names else ""
        lines.append(f"| `{k}` | {ft['dtype']} | {ft['shape']} | {ns} |")
    lines += [
        "",
        "## Episodes",
        "",
        "One episode per contiguous face-tracked run (>= 3 s) of a source clip, split into pieces of",
        "<= 20 s. `action[t] = observation.state[t+1]` inside a run (the run's last frame holds).",
        "Joint values are in the robot's own units (see the `names` and the robot's `ROBOT.md`).",
        "`observation.environment_state` duplicates the speech features + speaking flag under the key",
        "lerobot's ACT / Diffusion policies require when there is no camera.",
        "",
        "## Sources and licenses",
        "",
        "| clip | episodes | frames | license | source |",
        "|---|---|---|---|---|",
    ]
    for name, c in prov.get("clips", {}).items():
        src = f"[{c.get('title') or c.get('source')}]({c['source_url']})" if c.get("source_url") else str(c.get("source") or "")
        lines.append(f"| {name} | {c['episodes']} | {c['frames']} | {c.get('license')} | {src} |")
    lines += [
        "",
        "Source videos are public-domain or CC-BY works (license evidence is recorded per clip in the",
        "animacy human-motion dataset); the derived motion is released CC-BY-4.0 with attribution to the",
        "sources above. No no-derivatives or non-commercial material is included by construction.",
        "",
        "## meta/info.json",
        "",
        "```json",
        json.dumps(info, indent=4),
        "```",
        "",
    ]
    return "\n".join(lines)


def push_to_hub(out_dir: str, repo_id: str, private: bool = False, commit_message: Optional[str] = None) -> str:
    """Upload ``out_dir`` as a dataset repo and tag it ``v3.0`` (lerobot resolves that tag
    as the revision to download). Returns the dataset URL."""
    from huggingface_hub import HfApi

    with open(os.path.join(out_dir, INFO_PATH), encoding="utf-8") as fh:
        info = json.load(fh)
    prov_path = os.path.join(out_dir, PROVENANCE_PATH)
    prov = json.load(open(prov_path, encoding="utf-8")) if os.path.exists(prov_path) else {}
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(dataset_card(info, prov, repo_id))
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(folder_path=out_dir, repo_id=repo_id, repo_type="dataset",
                      commit_message=commit_message or f"animacy lerobot export ({info['total_episodes']} episodes, {info['total_frames']} frames)",
                      ignore_patterns=["images/", "*.lock", ".cache/**"])
    try:
        api.delete_tag(repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
    except Exception:
        pass
    api.create_tag(repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
    return f"https://huggingface.co/datasets/{repo_id}"
