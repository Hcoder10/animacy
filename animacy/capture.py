"""``animacy capture`` — webcam / video file -> canonical human motion clip.

Run
---
    animacy capture --source 0 -o data/clips/me --preview            # webcam, q to stop
    animacy capture --source data/raw/talk.webm -o data/clips/talk --duration 120
    animacy capture --source data/raw/ -o data/clips/               # every video -> <o>/<stem>/
    options: --arm right|left|none  --no-audio  --neutral-seconds 1.0 (0 = whole-clip median)

Output: ``<o>/motion.parquet`` (30 Hz canonical frames, smoothed), ``audio.wav``
(16 kHz mono, same clock), ``meta.json`` (source, license evidence copied from
``data/raw/sources.json`` when the video came from ``scripts/fetch_sources.py``,
neutral pose, tool versions, models, VAD/ffmpeg backends, validity stats).

Models
------
MediaPipe Tasks ``.task`` bundles are downloaded on first use from the official
Google storage bucket into ``<repo>/data/models/`` (override with
``ANIMACY_MODELS_DIR``): ``face_landmarker.task`` (478 landmarks + 52 ARKit-style
blendshapes + facial transformation matrix) and ``pose_landmarker_lite.task``
(33 world landmarks; ``ANIMACY_POSE_MODEL=full`` for the heavier one).

Pipeline per decoded frame
--------------------------
FaceLandmarker (VIDEO mode, monotonic ms timestamps) -> transformation matrix
-> ``capture_math.head_pose_from_matrix`` -> body-frame rotation + mm
translation; blendshapes -> gaze/brow/eye/mouth channels. PoseLandmarker world
landmarks -> ``pose_to_body`` -> torso lean/yaw and the puppet arm chain
(``--arm left`` mirrors y before the arm math). Every sample carries the source
timestamp (``CAP_PROP_POS_MSEC`` when monotonic, else ``frame_index / fps``);
after decoding, all channels are resampled onto a 30 Hz grid aligned with the
audio clock (``capture_math.resample_to_grid``: linear between valid
neighbours, never across a gap), head/torso channels are zeroed against the
neutral pose (median of the first ``--neutral-seconds`` of *valid* frames, or
of the whole clip when 0; zeroed: head 6-DoF, gaze, brows, torso; absolute:
eye_open, mouth_open, smile, arm), smoothed with a zero-phase 8 Hz Butterworth per
contiguous valid run, clipped to the schema's sanity bounds, and the
``speaking`` flag comes from VAD on the audio (``animacy.vad``: silero-vad if
installed, else energy + hysteresis; ``meta['vad']`` says which).

Sign derivation (the contract, docs/CANONICAL.md)
-------------------------------------------------
MediaPipe's facial transformation matrix lives in its "metric 3D" camera frame:
+X image-right, +Y up, +Z toward the viewer, centimetres, right-handed; the
matrix maps the canonical face model (nose along +Z, face's left along +X) into
it, so it is ~identity for a face looking into the camera. For an unmirrored
image the subject's left is the image right, hence body(x fwd, y left, z up) =
(cam_z, cam_x, cam_y), i.e. ``R_body = P R_cam P^T`` with
``P = [[0,0,1],[1,0,0],[0,1,0]]``. ZYX Euler angles of ``R_body`` give
yaw about +z (+ = turn left), pitch about +y (+ = nose DOWN, so
``head_pitch = -pitch``), roll about +x (+ = right ear drops). Translation
``head_xyz_mm = 10 * P (t - t_neutral)``. Full derivation with the axis
checks: ``animacy/capture_math.py`` module docstring.

Empirical verification (``scripts/capture_debug_frames.py`` writes annotated
frames to ``data/debug/``; each sign was read off real frames, and every
frame is also cross-checked against image-space landmark geometry that does
not depend on the matrix convention: nose-vs-cheek offset for yaw, eye-line
tilt for roll, nose-vs-ear height for pitch, face size for head_x, face
centre for head_y/z, iris offset within the eye for gaze_yaw). See the
capture report in the session notes for which frames confirmed what.

Known limitations
-----------------
* ``speaking`` is "any voice on the track": an off-camera interviewer counts.
* Torso lean uses MediaPipe's hip estimate; on head-and-shoulders footage the
  hips are extrapolated, so torso_lean_* is a coarse signal there.
* Arm channels need shoulder, elbow, wrist and the three hand landmarks all
  visible (``arm_valid``); talking-head footage rarely has them. Wrist roll /
  hand_open come from BlazePose's 3 hand points, which are noisy.
* Translation scale assumes MediaPipe's default virtual camera FOV, so
  head_x/y/z are proportional to true mm, not calibrated.
* Gaze from blendshapes saturates well before 40 deg; treat as qualitative.
* Neutral zeroing on downloaded video uses the whole-clip median
  (``--neutral-seconds 0``) because nobody poses neutral for the camera first.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import __version__
from . import capture_math as cm
from .schema import (ARM_CHANNELS, BOUNDS, CHANNELS, FACE_CHANNELS, RATE_HZ, TORSO_CHANNELS,
                     HumanClip, empty_frames)

MODEL_URLS = {
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "pose_landmarker_full.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
}
VIDEO_EXT = (".mp4", ".mkv", ".webm", ".mov", ".avi", ".ogv", ".mpg", ".mpeg", ".m4v")
SMOOTH_CUTOFF_HZ = 8.0
FACE_VALUE_KEYS = ["gaze_yaw", "gaze_pitch", "brow_l", "brow_r", "brow_furrow",
                   "eye_open_l", "eye_open_r", "mouth_open", "smile"]
POSE_VIS_THRESHOLD = 0.5


def models_dir() -> str:
    d = os.environ.get("ANIMACY_MODELS_DIR")
    if d:
        return d
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "models")


def ensure_model(name: str) -> str:
    """Path to a .task model, downloading it from Google's bucket on first use."""
    d = models_dir()
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        return path
    url = MODEL_URLS[name]
    print(f"downloading {name} from {url}")
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    return path


# ------------------------------------------------------------------ trackers
class Trackers:
    """Face + pose landmarkers in VIDEO mode (synchronous, timestamped)."""

    def __init__(self, want_pose: bool = True) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        self.face_model = "face_landmarker.task"
        self.pose_model = f"pose_landmarker_{os.environ.get('ANIMACY_POSE_MODEL', 'lite')}.task"
        self.face = mp_vision.FaceLandmarker.create_from_options(mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model(self.face_model)),
            output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=1,
            running_mode=mp_vision.RunningMode.VIDEO))
        self.pose = None
        if want_pose:
            self.pose = mp_vision.PoseLandmarker.create_from_options(mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=ensure_model(self.pose_model)),
                num_poses=1, running_mode=mp_vision.RunningMode.VIDEO))
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray, t_s: float, arm: str = "right") -> Dict:
        """One frame -> raw sample dict (absolute, un-zeroed values)."""
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(round(t_s * 1000)), self._last_ts + 1)  # VIDEO mode needs strictly increasing ms
        self._last_ts = ts
        s: Dict = {"t": t_s, "face_ok": False, "pose_ok": False, "arm_ok": False}
        try:
            fr = self.face.detect_for_video(img, ts)
        except Exception:
            fr = None
        if fr is not None and fr.face_landmarks and fr.facial_transformation_matrixes:
            m = np.array(fr.facial_transformation_matrixes[0], dtype=float).reshape(4, 4)
            r_body, t_mm = cm.head_pose_from_matrix(m)
            s["face_ok"] = True
            s["matrix"] = m
            s["head_angles"] = cm.head_angles_deg(r_body)
            s["head_trans"] = t_mm
            blend = {c.category_name: float(c.score) for c in fr.face_blendshapes[0]} if fr.face_blendshapes else {}
            s["blend"] = blend
            s["face_raw"] = cm.face_raw_from_blendshapes(blend)
            lm = np.array([[p.x, p.y] for p in fr.face_landmarks[0]], dtype=float)
            s["lm_xy"] = lm
            s["iris_gaze_u"] = cm.iris_gaze_yaw_unit(lm) if len(lm) >= 478 else float("nan")
        if self.pose is not None:
            try:
                pr = self.pose.detect_for_video(img, ts)
            except Exception:
                pr = None
            if pr is not None and pr.pose_world_landmarks:
                world = np.array([[p.x, p.y, p.z] for p in pr.pose_world_landmarks[0]], dtype=float)
                norm = pr.pose_landmarks[0]
                vis = np.array([(p.visibility if p.visibility is not None else 1.0) for p in norm], dtype=float)
                inside = np.array([0.0 <= p.x <= 1.0 and 0.0 <= p.y <= 1.0 for p in norm], dtype=bool)
                body = cm.pose_to_body(world)
                s["pose_body"] = body
                s["pose_vis"] = vis
                P = cm.POSE
                sh_ok = vis[P["l_shoulder"]] > POSE_VIS_THRESHOLD and vis[P["r_shoulder"]] > POSE_VIS_THRESHOLD
                if sh_ok:
                    s["pose_ok"] = True
                    s["torso_vals"] = cm.torso_channels(body)
                    s["hips_visible"] = bool(vis[P["l_hip"]] > POSE_VIS_THRESHOLD and vis[P["r_hip"]] > POSE_VIS_THRESHOLD)
                if arm in ("right", "left"):
                    pre = "l_" if arm == "left" else "r_"
                    need = [P[pre + k] for k in ("shoulder", "elbow", "wrist", "index", "pinky", "thumb")]
                    if all(vis[i] > POSE_VIS_THRESHOLD and inside[i] for i in need):
                        s["arm_ok"] = True
                        s["arm_vals"] = cm.arm_channels(body, side=arm)
        return s


# ------------------------------------------------------------------ overlay (preview + debug)
def draw_overlay(frame_bgr: np.ndarray, sample: Dict, rel: Optional[Dict] = None, title: str = "") -> np.ndarray:
    """Draw the sample's values (and neutral-relative channels if given) on a copy of the frame."""
    import cv2

    out = frame_bgr.copy()
    h, w = out.shape[:2]
    lines: List[str] = [title] if title else []
    if sample.get("face_ok"):
        ay, ap, ar = sample["head_angles"]
        tx, ty, tz = sample["head_trans"]
        lines.append(f"abs yaw {ay:+6.1f} pitch {ap:+6.1f} roll {ar:+6.1f}   t_mm x {tx:+6.0f} y {ty:+6.0f} z {tz:+6.0f}")
        if rel:
            lines.append(f"REL head_yaw {rel['head_yaw']:+6.1f} head_pitch {rel['head_pitch']:+6.1f} head_roll {rel['head_roll']:+6.1f}")
            lines.append(f"REL head_x {rel['head_x']:+6.0f} head_y {rel['head_y']:+6.0f} head_z {rel['head_z']:+6.0f} mm")
        fv = sample["face_raw"]
        lines.append(f"raw gaze_yaw {fv['gaze_yaw']:+5.1f} gaze_pitch {fv['gaze_pitch']:+5.1f}  iris_u {sample.get('iris_gaze_u', float('nan')):+.2f}")
        if rel:
            lines.append(f"REL gaze_yaw {rel['gaze_yaw']:+5.1f} gaze_pitch {rel['gaze_pitch']:+5.1f}  brow L {rel['brow_l']:.2f} R {rel['brow_r']:.2f} furrow {rel['brow_furrow']:.2f}")
        lines.append(f"raw brow L {fv['brow_l_signed']:+.2f} R {fv['brow_r_signed']:+.2f}  eye L {fv['eye_open_l']:.2f} R {fv['eye_open_r']:.2f}  mouth {fv['mouth_open']:.2f} smile {fv['smile']:.2f}")
        lm = sample.get("lm_xy")
        if lm is not None:
            for i in (1, 33, 263, 234, 454, 468, 473):  # nose, R/L eye outer, R/L cheek, R/L iris
                cv2.circle(out, (int(lm[i, 0] * w), int(lm[i, 1] * h)), 3, (0, 255, 255), -1)
            cv2.line(out, (int(lm[33, 0] * w), int(lm[33, 1] * h)), (int(lm[263, 0] * w), int(lm[263, 1] * h)), (0, 200, 255), 1)
    else:
        lines.append("face: not detected")
    if sample.get("pose_ok"):
        tv = sample["torso_vals"]
        lines.append(f"torso lean_fwd {tv['torso_lean_fwd']:+5.1f} lean_side {tv['torso_lean_side']:+5.1f} yaw {tv['torso_yaw']:+5.1f}  hips_vis {int(sample.get('hips_visible', False))}")
    if sample.get("arm_ok"):
        av = sample["arm_vals"]
        lines.append(f"arm sh_yaw {av['shoulder_yaw']:+5.0f} sh_pitch {av['shoulder_pitch']:5.0f} elbow {av['elbow_flex']:5.0f} wr_roll {av['wrist_roll']:+5.0f} wr_pitch {av['wrist_pitch']:+5.0f} open {av['hand_open']:.2f}")
    if "speaking" in sample:
        lines.append(f"speaking {int(sample['speaking'])}")
    y = 18
    for ln in lines:
        cv2.putText(out, ln, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, ln, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 17
    return out


# ------------------------------------------------------------------ sources
def open_source(source: str):
    """(cv2.VideoCapture, is_webcam)."""
    import cv2

    if source.isdigit():
        idx = int(source)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap, True
    cap = cv2.VideoCapture(source)
    return cap, False


def list_videos(path: str) -> List[str]:
    if os.path.isdir(path):
        return sorted(os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(VIDEO_EXT))
    return [path]


def source_record(video_path: str) -> Dict:
    """License record from ``sources.json`` next to the video, if any."""
    idx = os.path.join(os.path.dirname(os.path.abspath(video_path)), "sources.json")
    if not os.path.exists(idx):
        return {}
    try:
        for rec in json.load(open(idx, encoding="utf-8")):
            if rec.get("file") == os.path.basename(video_path):
                return rec
    except Exception:
        pass
    return {}


# ------------------------------------------------------------------ run
def run_source(source: str, arm: str, duration: float, preview: bool, want_audio: bool,
               on_sample=None) -> Tuple[List[Dict], Optional[np.ndarray], Dict]:
    """Decode ``source``, run the trackers, return (samples, audio, info)."""
    import cv2

    from .audio import MicRecorder, extract_audio
    from .schema import AUDIO_SR

    cap, is_cam = open_source(source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source {source!r}")
    trackers = Trackers(want_pose=True)
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not is_cam and src_fps <= 0:
        src_fps = 30.0
    proc_hz = min(src_fps, 30.0) if src_fps > 0 else 30.0
    info: Dict = {"source": "webcam" if is_cam else "video", "src_fps": src_fps,
                  "face_model": trackers.face_model, "pose_model": trackers.pose_model,
                  "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    audio: Optional[np.ndarray] = None
    mic = None
    if is_cam and want_audio:
        try:
            mic = MicRecorder(sr=AUDIO_SR)
            mic.start()
            info["audio_backend"] = "sounddevice mic"
        except Exception as exc:
            print(f"mic unavailable ({type(exc).__name__}: {exc}); recording without audio")
            mic = None
            info["audio_backend"] = "none"

    samples: List[Dict] = []
    idx, last_pos, next_proc, t0_cam = 0, -1.0, 0.0, None
    n_src = 0
    t_start_wall = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n_src += 1
            if is_cam:
                now = time.perf_counter()
                if t0_cam is None:
                    t0_cam = now
                t = now - t0_cam
            else:
                pos = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
                t_idx = idx / src_fps
                # trust the container timestamp only when it is monotonic and near the index-based time
                t = pos if (pos > last_pos and abs(pos - t_idx) < 0.5) else t_idx
                last_pos = max(last_pos, pos)
            idx += 1
            if duration > 0 and t > duration:
                break
            if not is_cam and t + 1e-6 < next_proc:
                continue  # decimate >30 fps sources
            next_proc = t + 1.0 / proc_hz - 1e-6
            s = trackers.detect(frame, t, arm=arm)
            s["frame_idx"] = idx - 1
            samples.append(s)
            if on_sample is not None:
                on_sample(frame, s)
            if preview:
                cv2.imshow("animacy capture (q to stop)", draw_overlay(frame, s, title=f"t={t:6.2f}s"))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if len(samples) % 300 == 0:
                nf = sum(1 for x in samples if x["face_ok"])
                print(f"  {len(samples)} frames, t={t:.1f}s, face {nf}/{len(samples)}", flush=True)
    finally:
        cap.release()
        if preview:
            cv2.destroyAllWindows()
    info["n_src_frames"] = n_src
    info["wall_s"] = time.perf_counter() - t_start_wall
    if not samples:
        return samples, None, info
    clip_len = samples[-1]["t"]
    if is_cam:
        if mic is not None:
            raw = mic.stop()
            audio = mic.slice(raw, t0_cam, clip_len + 1.0 / RATE_HZ)
    elif want_audio:
        audio, backend = extract_audio(source, AUDIO_SR, max_seconds=(duration if duration > 0 else 0.0))
        info["audio_backend"] = backend
        if audio is None:
            print(f"no audio extracted: {backend}")
    else:
        info["audio_backend"] = "disabled"
    return samples, audio, info


# ------------------------------------------------------------------ assemble
def neutral_window(samples: List[Dict], key_ok: str, neutral_seconds: float) -> List[int]:
    ok = [i for i, s in enumerate(samples) if s.get(key_ok)]
    if not ok or neutral_seconds <= 0:
        return ok
    t0 = samples[ok[0]]["t"]
    return [i for i in ok if samples[i]["t"] <= t0 + neutral_seconds]


def build_frames(samples: List[Dict], neutral_seconds: float, rate_hz: float = RATE_HZ,
                 duration: Optional[float] = None) -> Tuple[pd.DataFrame, Dict]:
    """Raw samples -> canonical 30 Hz frame table (no ``speaking`` yet) + neutral/stats."""
    t_src = np.array([s["t"] for s in samples], dtype=float)
    face_ok = np.array([s["face_ok"] for s in samples], dtype=bool)
    pose_ok = np.array([s.get("pose_ok", False) for s in samples], dtype=bool)
    arm_ok = np.array([s.get("arm_ok", False) for s in samples], dtype=bool)

    # neutral head pose
    nwin = neutral_window(samples, "face_ok", neutral_seconds)
    neutral = cm.neutral_pose(np.array([samples[i]["head_angles"] for i in nwin]).reshape(-1, 3),
                              np.array([samples[i]["head_trans"] for i in nwin]).reshape(-1, 3))
    neutral["n_frames"] = len(nwin)
    neutral["seconds"] = float(neutral_seconds)
    twin = neutral_window(samples, "pose_ok", neutral_seconds)
    tn = np.array([[samples[i]["torso_vals"][c] for c in TORSO_CHANNELS] for i in twin]).reshape(-1, 3)
    neutral["torso_deg"] = np.median(tn, axis=0).tolist() if len(tn) else [0.0, 0.0, 0.0]
    neutral["face_raw"] = ({k: float(np.median([samples[i]["face_raw"][k] for i in nwin])) for k in cm.RAW_FACE_KEYS}
                           if nwin else {})

    # absolute -> neutral-relative per raw sample
    face_rows = np.full((len(samples), len(FACE_CHANNELS)), np.nan)
    for i, s in enumerate(samples):
        if not s["face_ok"]:
            continue
        r_body = cm.head_rotmat_from_angles_deg(*s["head_angles"])
        yaw, pitch, roll, x, y, z = cm.relative_head(r_body, s["head_trans"], neutral)
        fv = cm.face_channels_relative(s["face_raw"], neutral["face_raw"])
        face_rows[i] = [yaw, pitch, roll, x, y, z] + [fv[k] for k in FACE_VALUE_KEYS]
    torso_rows = np.full((len(samples), 3), np.nan)
    for i, s in enumerate(samples):
        if s.get("pose_ok"):
            torso_rows[i] = [s["torso_vals"][c] - n for c, n in zip(TORSO_CHANNELS, neutral["torso_deg"])]
    arm_rows = np.full((len(samples), len(ARM_CHANNELS)), np.nan)
    for i, s in enumerate(samples):
        if s.get("arm_ok"):
            arm_rows[i] = [s["arm_vals"][c] for c in ARM_CHANNELS]

    # resample each group onto the grid, smooth per run, clip to bounds
    if duration is None:
        duration = float(t_src[-1])
    t_grid, face_g, face_v = cm.resample_to_grid(t_src, face_rows, face_ok, rate_hz, duration)
    _, torso_g, torso_v = cm.resample_to_grid(t_src, torso_rows, pose_ok, rate_hz, duration)
    _, arm_g, arm_v = cm.resample_to_grid(t_src, arm_rows, arm_ok, rate_hz, duration)
    face_g = cm.smooth_runs(face_g, face_v, SMOOTH_CUTOFF_HZ, rate_hz)
    torso_g = cm.smooth_runs(torso_g, torso_v, SMOOTH_CUTOFF_HZ, rate_hz)
    arm_g = cm.smooth_runs(arm_g, arm_v, SMOOTH_CUTOFF_HZ, rate_hz)

    df = empty_frames(len(t_grid), rate_hz)
    df["t"] = t_grid.astype(np.float32)
    for j, c in enumerate(FACE_CHANNELS):
        df[c] = face_g[:, j]
    for j, c in enumerate(TORSO_CHANNELS):
        df[c] = torso_g[:, j]
    for j, c in enumerate(ARM_CHANNELS):
        df[c] = arm_g[:, j]
    for c in FACE_CHANNELS + TORSO_CHANNELS + ARM_CHANNELS:
        lo, hi = BOUNDS[c]
        df[c] = np.clip(df[c].to_numpy(dtype=float), lo, hi)
    # invalid groups carry NaN, never a fake zero
    df.loc[~face_v, FACE_CHANNELS] = np.nan
    df.loc[~torso_v, TORSO_CHANNELS] = np.nan
    df.loc[~arm_v, ARM_CHANNELS] = np.nan
    df["face_valid"] = face_v.astype(np.float32)
    df["arm_valid"] = arm_v.astype(np.float32)
    df["speaking"] = 0.0
    df = df[CHANNELS].astype(np.float32)

    stats = {
        "n_raw_samples": int(len(samples)),
        "n_frames": int(len(df)),
        "face_valid_frac": float(face_v.mean()) if len(face_v) else 0.0,
        "torso_valid_frac": float(torso_v.mean()) if len(torso_v) else 0.0,
        "arm_valid_frac": float(arm_v.mean()) if len(arm_v) else 0.0,
        "hips_visible_frac": float(np.mean([bool(s.get("hips_visible", False)) for s in samples if s.get("pose_ok")])) if pose_ok.any() else 0.0,
    }
    for c in ("head_yaw", "head_pitch", "head_roll", "head_x", "head_y", "head_z", "gaze_yaw", "mouth_open"):
        v = df[c].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        stats[f"{c}_std"] = float(v.std()) if len(v) else float("nan")
        stats[f"{c}_p05_p95"] = [float(np.percentile(v, 5)), float(np.percentile(v, 95))] if len(v) else [float("nan")] * 2
    return df, {"neutral": neutral, "stats": stats}


def tool_versions() -> Dict[str, str]:
    out = {"animacy": __version__, "python": sys.version.split()[0]}
    for mod in ("mediapipe", "cv2", "numpy", "scipy", "torch"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    return out


def capture_one(source: str, output: str, arm: str = "right", duration: float = 0.0, no_audio: bool = False,
                preview: bool = False, neutral_seconds: float = 1.0) -> HumanClip:
    from .schema import AUDIO_SR
    from .vad import speaking_mask

    samples, audio, info = run_source(source, arm, duration, preview, want_audio=not no_audio)
    if not samples:
        raise SystemExit("no frames decoded")
    is_cam = info["source"] == "webcam"
    frames, extra = build_frames(samples, neutral_seconds, RATE_HZ)
    n = len(frames)
    if audio is not None:
        want = int(round(n / RATE_HZ * AUDIO_SR))  # audio spans the last frame's period too
        audio = np.pad(audio[:want], (0, max(0, want - len(audio))))
    speaking, vad_backend = speaking_mask(audio, AUDIO_SR, frames["t"].to_numpy(dtype=float))
    frames["speaking"] = speaking.astype(np.float32)
    extra["stats"]["speaking_frac"] = float(speaking.mean()) if n else 0.0

    src_rec = {} if is_cam else source_record(source)
    subject = "self" if is_cam else hashlib.sha1(os.path.basename(source).encode()).hexdigest()[:10]
    meta = {
        "source": info["source"],
        "source_path": None if is_cam else os.path.abspath(source),
        "source_url": src_rec.get("page_url") or src_rec.get("url"),
        "source_file_url": src_rec.get("url"),
        "title": src_rec.get("title"),
        "license": "self" if is_cam else src_rec.get("license", "UNKNOWN"),
        "license_evidence": src_rec.get("license_evidence"),
        "artist": src_rec.get("artist"),
        "rate_hz": RATE_HZ,
        "subject": subject,
        "arm": arm,
        "neutral": extra["neutral"],
        "tool_versions": tool_versions(),
        "models": {"face": info["face_model"], "pose": info["pose_model"], "models_dir": models_dir()},
        "vad": vad_backend,
        "audio_backend": info.get("audio_backend", "none"),
        "src_fps": info["src_fps"], "src_size": [info["width"], info["height"]], "n_src_frames": info["n_src_frames"],
        "smoothing": {"kind": "zero-phase butterworth order 2 per contiguous valid run", "cutoff_hz": SMOOTH_CUTOFF_HZ},
        "sign_convention": "docs/CANONICAL.md; mapping derived in animacy/capture_math.py, verified on real video 2026-08-26",
        "stats": extra["stats"],
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    clip = HumanClip.from_frames(frames, **meta)
    clip.audio = audio
    clip.sr = AUDIO_SR
    clip.save(output)
    return clip


def report(clip: HumanClip, output: str) -> None:
    st = clip.meta["stats"]
    probs = clip.validate()
    print(f"wrote {output}: {len(clip)} frames ({clip.duration:.1f}s) validate={'OK' if not probs else probs}")
    print(f"  face_valid {st['face_valid_frac']:.1%}  arm_valid {st['arm_valid_frac']:.1%}  torso_valid {st['torso_valid_frac']:.1%}"
          f"  speaking {st.get('speaking_frac', 0):.1%}  [vad={clip.meta['vad']}, audio={clip.meta['audio_backend']}]")
    print(f"  head_yaw std {st['head_yaw_std']:.1f} deg  head_pitch std {st['head_pitch_std']:.1f} deg  head_roll std {st['head_roll_std']:.1f} deg"
          f"  head_x std {st['head_x_std']:.0f} mm")


def main(args) -> int:
    """Entry point wired from ``animacy.cli`` (args: source, output, arm, duration, no_audio, preview, neutral_seconds)."""
    source = str(args.source)
    outputs: List[Tuple[str, str]] = []
    if not source.isdigit() and os.path.isdir(source):
        vids = list_videos(source)
        if not vids:
            print(f"no videos in {source}")
            return 1
        outputs = [(v, os.path.join(args.output, os.path.splitext(os.path.basename(v))[0])) for v in vids]
    else:
        outputs = [(source, args.output)]
    rc = 0
    for src, out in outputs:
        print(f"== {src} -> {out}")
        try:
            clip = capture_one(src, out, arm=args.arm, duration=args.duration, no_audio=args.no_audio,
                               preview=args.preview, neutral_seconds=args.neutral_seconds)
        except SystemExit as exc:
            print(f"   FAILED: {exc}")
            rc = 1
            continue
        report(clip, out)
        if clip.validate():
            rc = 1
    return rc
