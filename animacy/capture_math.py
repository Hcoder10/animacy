"""Pure-numpy geometry for the capture stage (no camera, no MediaPipe).

Everything that turns MediaPipe outputs into canonical channels lives here so
it can be unit-tested with synthetic inputs (``tests/test_capture_math.py``).
``animacy.capture`` only decodes frames, runs the models, and calls into this.

Frames
------
* **cam** — MediaPipe Face Landmarker "metric 3D" frame: origin at the camera,
  +X to the *image right*, +Y up, +Z toward the viewer (out of the screen),
  right-handed, centimetres. The facial transformation matrix ``M`` maps the
  canonical face model (nose along +Z, face's left along +X, up along +Y) into
  that frame, so ``M[:3,:3] ~ I`` for a face looking straight into the camera.
* **body** — the canonical frame of ``docs/CANONICAL.md``: +x forward (toward
  the camera, from the subject's point of view), +y the subject's left, +z up.

For an unmirrored image, the subject's left is on the image right, so at the
neutral pose ``body_x = cam_z``, ``body_y = cam_x``, ``body_z = cam_y``:

    P = [[0, 0, 1],
         [1, 0, 0],
         [0, 1, 0]]          v_body = P @ v_cam,   det(P) = +1

A rotation ``R_cam`` (face -> cam) expressed in body coordinates is
``R_body = P @ R_cam @ P.T``. Its ZYX Euler angles (yaw about z, pitch about
y, roll about x, ``R = Rz(yaw) Ry(pitch) Rx(roll)``) then have these signs:

* yaw about +z (up): +x (forward) rotates toward +y (left) -> **turn left = +**
  (matches ``head_yaw``).
* pitch about +y (left): +z (up) rotates toward +x (forward), i.e. the nose
  drops -> ROS pitch + = look DOWN, so **head_pitch = -pitch**.
* roll about +x (forward): +y (left) rotates toward +z (up): the left side
  rises, the right ear drops toward the right shoulder -> **head_roll = +roll**.

Translation: ``head_x/y/z = 10 * P @ (t_cam - t_cam_neutral)`` (cm -> mm).
A face in front of the camera sits at negative cam-z; approaching the camera
makes z less negative, so +head_x = toward the camera.

Pose Landmarker world landmarks use the *image* orientation (x image-right,
y **down**, z away from the camera, metres, origin at the hip midpoint), so
``body = (-z, x, -y)`` (``pose_to_body``).

These derivations were verified empirically on real video by
``scripts/capture_debug_frames.py`` (annotated frames in ``data/debug/``); see
the module docstring of ``animacy.capture`` for what was checked.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# cam (MediaPipe metric 3D) -> body (canonical): x_b = z_c, y_b = x_c, z_b = y_c
CAM_TO_BODY = np.array([[0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0]])

# Pose world landmarks (x right, y down, z away) -> body (x fwd, y left, z up)
POSE_TO_BODY = np.array([[0.0, 0.0, -1.0],
                         [1.0, 0.0, 0.0],
                         [0.0, -1.0, 0.0]])

# MediaPipe BlazePose landmark indices.
POSE = {
    "nose": 0,
    "l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16, "l_pinky": 17, "r_pinky": 18,
    "l_index": 19, "r_index": 20, "l_thumb": 21, "r_thumb": 22,
    "l_hip": 23, "r_hip": 24,
}

# Blendshape -> gaze scale. ARKit-style eyeLook* coefficients reach ~1 at the
# extreme of eye travel, roughly 30 deg horizontally and 20 deg vertically.
GAZE_YAW_DEG_PER_UNIT = 30.0
GAZE_PITCH_DEG_PER_UNIT = 20.0


# --------------------------------------------------------------------- rotations
def euler_zyx_to_rotmat(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` (radians)."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def rotmat_to_euler_zyx(r: np.ndarray) -> Tuple[float, float, float]:
    """Inverse of :func:`euler_zyx_to_rotmat`: (yaw, pitch, roll) in radians."""
    sp = float(np.clip(-r[2, 0], -1.0, 1.0))
    pitch = np.arcsin(sp)
    if abs(sp) < 0.99999:
        roll = np.arctan2(r[2, 1], r[2, 2])
        yaw = np.arctan2(r[1, 0], r[0, 0])
    else:  # gimbal lock: pin roll, put everything in yaw
        roll = 0.0
        yaw = np.arctan2(-r[0, 1], r[1, 1])
    return float(yaw), float(pitch), float(roll)


def rotation_about(axis: Sequence[float], deg: float) -> np.ndarray:
    """Rodrigues rotation matrix about a unit axis (for tests)."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    th = np.radians(deg)
    return np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * (k @ k)


# --------------------------------------------------------------------- head pose
def head_pose_from_matrix(m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """4x4 MediaPipe facial transformation matrix -> (R_body, t_body_mm)."""
    m = np.asarray(m, dtype=float).reshape(4, 4)
    r_body = CAM_TO_BODY @ m[:3, :3] @ CAM_TO_BODY.T
    t_body_mm = CAM_TO_BODY @ m[:3, 3] * 10.0
    return r_body, t_body_mm


def head_angles_deg(r_body: np.ndarray) -> Tuple[float, float, float]:
    """(head_yaw, head_pitch, head_roll) in canonical signs, degrees."""
    yaw, pitch, roll = rotmat_to_euler_zyx(r_body)
    return float(np.degrees(yaw)), float(-np.degrees(pitch)), float(np.degrees(roll))


def head_rotmat_from_angles_deg(head_yaw: float, head_pitch: float, head_roll: float) -> np.ndarray:
    """Inverse of :func:`head_angles_deg` (canonical signs in, R_body out)."""
    return euler_zyx_to_rotmat(np.radians(head_yaw), np.radians(-head_pitch), np.radians(head_roll))


def neutral_pose(angles_deg: np.ndarray, trans_mm: np.ndarray) -> Dict[str, List[float]]:
    """Median head angles/translation over the neutral window (rows = valid frames)."""
    angles_deg = np.asarray(angles_deg, dtype=float).reshape(-1, 3)
    trans_mm = np.asarray(trans_mm, dtype=float).reshape(-1, 3)
    if len(angles_deg) == 0:
        return {"head_angles_deg": [0.0, 0.0, 0.0], "head_trans_mm": [0.0, 0.0, 0.0]}
    return {"head_angles_deg": np.median(angles_deg, axis=0).tolist(),
            "head_trans_mm": np.median(trans_mm, axis=0).tolist()}


def relative_head(r_body: np.ndarray, t_body_mm: np.ndarray, neutral: Dict) -> Tuple[float, float, float, float, float, float]:
    """Head pose relative to the neutral pose: (yaw, pitch, roll, x, y, z).

    Rotation is ``R_neutral^T @ R`` (angles about the *neutral head's* axes,
    which for a subject facing the camera coincide with the body axes);
    translation is a plain difference in body coordinates.
    """
    r_n = head_rotmat_from_angles_deg(*neutral["head_angles_deg"])
    yaw, pitch, roll = head_angles_deg(r_n.T @ r_body)
    d = np.asarray(t_body_mm, dtype=float) - np.asarray(neutral["head_trans_mm"], dtype=float)
    return yaw, pitch, roll, float(d[0]), float(d[1]), float(d[2])


# --------------------------------------------------------------------- small-face crop path
# MediaPipe's face-geometry pipeline models the image as a pinhole camera with this
# vertical FOV (face_geometry "perspective camera" default). The crop path treats a
# square crop as the whole image, so its metric outputs are in the crop's virtual
# camera; these helpers map them back to the full frame's virtual camera.
FACE_GEOMETRY_VFOV_DEG = 63.0


def crop_box_for_face(cx: float, cy: float, size: float, width: int, height: int,
                      margin: float = 2.5, min_side: int = 192) -> Tuple[int, int, int]:
    """Square crop (x0, y0, side) around a face centre/size in pixels, clamped to the image."""
    side = int(round(max(size * margin, min_side)))
    side = min(side, width, height)
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x0 = min(max(x0, 0), width - side)
    y0 = min(max(y0, 0), height - side)
    return x0, y0, side


def crop_landmarks_to_full(lm_xy: np.ndarray, crop_box: Tuple[int, int, int], full_wh: Tuple[int, int]) -> np.ndarray:
    """Normalized landmarks of the crop -> normalized landmarks of the full frame."""
    x0, y0, side = crop_box
    w, h = full_wh
    out = np.array(lm_xy, dtype=float, copy=True)
    out[:, 0] = (x0 + out[:, 0] * side) / w
    out[:, 1] = (y0 + out[:, 1] * side) / h
    return out


def crop_to_full_translation(t_cam_cm: np.ndarray, crop_box: Tuple[int, int, int], full_wh: Tuple[int, int],
                             vfov_deg: float = FACE_GEOMETRY_VFOV_DEG) -> np.ndarray:
    """Translation from the crop's virtual camera -> the full frame's virtual camera (cm).

    Pinhole with focal length f = (H/2)/tan(vfov/2) in pixels: the crop's camera has
    f_crop = f * side/H, so ``z_full = z_crop * H/side``; the lateral position gains the
    crop centre's offset from the image centre, ``x_full = x_crop + du * |z_crop| / f_crop``
    (image-right = +x, image-down = -y). Rotation is unchanged: the crop path reports the
    head's orientation relative to the camera->face ray instead of the optical axis, a
    constant for a seated subject that the neutral zeroing removes.
    """
    x0, y0, side = crop_box
    w, h = full_wh
    t = np.asarray(t_cam_cm, dtype=float)
    f_crop = (side / 2.0) / np.tan(np.radians(vfov_deg) / 2.0)
    du = (x0 + side / 2.0) - w / 2.0
    dv = (y0 + side / 2.0) - h / 2.0
    dist = abs(float(t[2]))
    return np.array([t[0] + du * dist / f_crop, t[1] - dv * dist / f_crop, t[2] * h / side])


# --------------------------------------------------------------------- blendshapes
RAW_FACE_KEYS = ["gaze_yaw", "gaze_pitch", "brow_l_signed", "brow_r_signed",
                 "eye_open_l", "eye_open_r", "mouth_open", "smile"]


def face_raw_from_blendshapes(b: Dict[str, float]) -> Dict[str, float]:
    """ARKit-style blendshape scores (0..1, MediaPipe names) -> raw face signals.

    Raw = not yet neutral-zeroed (see :func:`face_channels_relative`). Left/Right
    in the names are the *subject's* left/right (ARKit convention; verified
    against iris-landmark geometry, see ``animacy.capture``). Gaze: for the
    left eye "Out" is toward the subject's left; for the right eye "In"
    (toward the nose) is toward the subject's left. ``brow_*_signed`` is
    raise minus furrow, because MediaPipe's brow baselines are strongly
    person-biased (a speaker can idle at browDown=0.67, browUp=0.00).
    """
    g = b.get
    gaze_yaw_u = 0.5 * ((g("eyeLookOutLeft", 0.0) - g("eyeLookInLeft", 0.0))
                        + (g("eyeLookInRight", 0.0) - g("eyeLookOutRight", 0.0)))
    gaze_pitch_u = 0.5 * ((g("eyeLookUpLeft", 0.0) - g("eyeLookDownLeft", 0.0))
                          + (g("eyeLookUpRight", 0.0) - g("eyeLookDownRight", 0.0)))
    inner = g("browInnerUp", 0.0)
    return {
        "gaze_yaw": float(gaze_yaw_u * GAZE_YAW_DEG_PER_UNIT),
        "gaze_pitch": float(gaze_pitch_u * GAZE_PITCH_DEG_PER_UNIT),
        "brow_l_signed": float(0.5 * (g("browOuterUpLeft", 0.0) + inner) - g("browDownLeft", 0.0)),
        "brow_r_signed": float(0.5 * (g("browOuterUpRight", 0.0) + inner) - g("browDownRight", 0.0)),
        "eye_open_l": float(np.clip(1.0 - g("eyeBlinkLeft", 0.0), 0, 1)),
        "eye_open_r": float(np.clip(1.0 - g("eyeBlinkRight", 0.0), 0, 1)),
        "mouth_open": float(np.clip(g("jawOpen", 0.0), 0, 1)),
        "smile": float(np.clip(0.5 * (g("mouthSmileLeft", 0.0) + g("mouthSmileRight", 0.0)), 0, 1)),
    }


def face_channels_relative(raw: Dict[str, float], neutral: Dict[str, float]) -> Dict[str, float]:
    """Raw face signals -> canonical channels, zeroed against the neutral medians.

    Neutral-relative: gaze (looking at the camera = 0) and brows (brow_l/r =
    positive part of the raise-minus-furrow change, brow_furrow = negative
    part, averaged). Absolute (natural zero): eye_open, mouth_open, smile —
    subtracting a talking clip's median mouth opening would erase speech.
    """
    d_l = raw["brow_l_signed"] - neutral.get("brow_l_signed", 0.0)
    d_r = raw["brow_r_signed"] - neutral.get("brow_r_signed", 0.0)
    return {
        "gaze_yaw": float(np.clip(raw["gaze_yaw"] - neutral.get("gaze_yaw", 0.0), -40, 40)),
        "gaze_pitch": float(np.clip(raw["gaze_pitch"] - neutral.get("gaze_pitch", 0.0), -30, 30)),
        "brow_l": float(np.clip(d_l, 0, 1)),
        "brow_r": float(np.clip(d_r, 0, 1)),
        "brow_furrow": float(np.clip(-0.5 * (d_l + d_r), 0, 1)),
        "eye_open_l": raw["eye_open_l"],
        "eye_open_r": raw["eye_open_r"],
        "mouth_open": raw["mouth_open"],
        "smile": raw["smile"],
    }


def iris_gaze_yaw_unit(lm_xy: np.ndarray) -> float:
    """Independent gaze-yaw estimate from iris landmarks (+ = subject's left = image right).

    ``lm_xy`` is the [478, 2] normalized face-mesh landmark array. Left eye:
    corners 362 (inner) / 263 (outer), iris centre 473; right eye: 133 (inner)
    / 33 (outer), iris centre 468. Returns the mean iris offset from the eye
    centre as a fraction of the half eye width, positive toward image right.
    """
    def offset(inner: int, outer: int, iris: int) -> float:
        c = 0.5 * (lm_xy[inner] + lm_xy[outer])
        half = 0.5 * abs(lm_xy[outer, 0] - lm_xy[inner, 0]) + 1e-6
        return float((lm_xy[iris, 0] - c[0]) / half)
    return 0.5 * (offset(362, 263, 473) + offset(133, 33, 468))


# --------------------------------------------------------------------- pose
def pose_to_body(world_xyz: np.ndarray) -> np.ndarray:
    """[N,3] pose world landmarks -> body frame (metres)."""
    return np.asarray(world_xyz, dtype=float) @ POSE_TO_BODY.T


def mirror_left_right(body_xyz: np.ndarray) -> np.ndarray:
    """Reflect y so the subject's left arm becomes a right arm (for ``--arm left``).

    A reflection flips handedness, so cross-product-based quantities (the
    hand-plane normal) are recomputed *after* mirroring in :func:`arm_channels`.
    """
    out = np.array(body_xyz, dtype=float, copy=True)
    out[:, 1] *= -1.0
    return out


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v * 0.0


def torso_channels(body_xyz: np.ndarray) -> Dict[str, float]:
    """torso_lean_fwd / torso_lean_side / torso_yaw (deg) from shoulder-hip geometry.

    Spine = hip midpoint -> shoulder midpoint. lean_fwd is its tilt toward +x
    (camera), lean_side its tilt toward +y (subject's left). torso_yaw is the
    heading of the right->left shoulder line about +z (0 when it points along
    +y; turning left rotates it toward -x).
    """
    p = body_xyz
    sh_l, sh_r = p[POSE["l_shoulder"]], p[POSE["r_shoulder"]]
    hip_l, hip_r = p[POSE["l_hip"]], p[POSE["r_hip"]]
    spine = 0.5 * (sh_l + sh_r) - 0.5 * (hip_l + hip_r)
    up = max(float(spine[2]), 1e-6)
    lean_fwd = np.degrees(np.arctan2(spine[0], up))
    lean_side = np.degrees(np.arctan2(spine[1], up))
    d = sh_l - sh_r
    yaw = np.degrees(np.arctan2(-d[0], d[1]))
    return {"torso_lean_fwd": float(lean_fwd), "torso_lean_side": float(lean_side), "torso_yaw": float(yaw)}


def arm_channels(body_xyz: np.ndarray, side: str = "right") -> Dict[str, float]:
    """Puppet-arm chain from pose world landmarks (body frame, metres).

    ``side='left'`` mirrors the left arm into the right-arm channels. Angles are
    absolute (a hanging arm is ``shoulder_pitch=0``), not neutral-zeroed.

    * shoulder_pitch: angle between the upper arm and straight down
      (0 hanging, 90 horizontal, 180 up).
    * shoulder_yaw: heading of the upper arm's horizontal projection
      (0 forward, + toward the subject's left), faded to 0 when the arm hangs
      (the heading of a vertical arm is undefined).
    * elbow_flex: 180 - interior elbow angle.
    * wrist_pitch: hand deviation from the forearm line, + when the hand tips up.
    * wrist_roll: thumb direction about the forearm axis, 0 = thumb up,
      + = thumb toward the subject's left (pronation of a right arm).
    * hand_open: finger extension (wrist->index vs forearm length) blended with
      index-pinky spread, 0 fist .. 1 spread.
    """
    p = mirror_left_right(body_xyz) if side == "left" else np.asarray(body_xyz, dtype=float)
    pre = "l_" if side == "left" else "r_"
    sh, el, wr = p[POSE[pre + "shoulder"]], p[POSE[pre + "elbow"]], p[POSE[pre + "wrist"]]
    idx, pky, thb = p[POSE[pre + "index"]], p[POSE[pre + "pinky"]], p[POSE[pre + "thumb"]]

    u = el - sh
    un = _unit(u)
    shoulder_pitch = np.degrees(np.arccos(np.clip(-un[2], -1.0, 1.0)))
    horiz = float(np.hypot(un[0], un[1]))
    yaw_gain = float(np.clip((horiz - 0.15) / 0.25, 0.0, 1.0))
    shoulder_yaw = np.degrees(np.arctan2(un[1], un[0])) * yaw_gain

    f = wr - el
    fn = _unit(f)
    cos_e = float(np.clip(np.dot(_unit(sh - el), fn), -1.0, 1.0))
    elbow_flex = 180.0 - np.degrees(np.arccos(cos_e))

    hand = 0.5 * (idx + pky) - wr
    hn = _unit(hand)
    perp = hand - np.dot(hand, fn) * fn
    ang = np.degrees(np.arccos(np.clip(np.dot(hn, fn), -1.0, 1.0)))
    # "up" reference perpendicular to the forearm; fall back to +y when the forearm is vertical
    z = np.array([0.0, 0.0, 1.0])
    e_up = z - np.dot(z, fn) * fn
    if np.linalg.norm(e_up) < 1e-6:
        e_up = np.array([0.0, 1.0, 0.0]) - np.dot(np.array([0.0, 1.0, 0.0]), fn) * fn
    e_up = _unit(e_up)
    e_left = np.cross(e_up, fn)
    wrist_pitch = ang * (1.0 if np.dot(perp, e_up) >= 0 else -1.0)

    tv = thb - 0.5 * (idx + pky)
    tv = tv - np.dot(tv, fn) * fn
    wrist_roll = np.degrees(np.arctan2(np.dot(tv, e_left), np.dot(tv, e_up))) if np.linalg.norm(tv) > 1e-6 else 0.0

    fore = float(np.linalg.norm(f)) + 1e-6
    ext = float(np.linalg.norm(idx - wr)) / fore          # ~0.35 fist .. ~0.7 open
    spread = float(np.linalg.norm(idx - pky)) / fore      # ~0.1 fist .. ~0.35 open
    hand_open = 0.5 * (np.clip((ext - 0.35) / 0.35, 0, 1) + np.clip((spread - 0.1) / 0.25, 0, 1))

    return {
        "shoulder_yaw": float(np.clip(shoulder_yaw, -90, 90)),
        "shoulder_pitch": float(np.clip(shoulder_pitch, -30, 180)),
        "elbow_flex": float(np.clip(elbow_flex, 0, 150)),
        "wrist_roll": float(np.clip(wrist_roll, -90, 90)),
        "wrist_pitch": float(np.clip(wrist_pitch, -80, 80)),
        "hand_open": float(np.clip(hand_open, 0, 1)),
    }


# --------------------------------------------------------------------- time series
def contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """[(start, stop), ...] half-open index ranges where ``mask`` is True."""
    mask = np.asarray(mask, dtype=bool)
    runs, start = [], None
    for i in range(len(mask) + 1):
        inside = i < len(mask) and mask[i]
        if inside and start is None:
            start = i
        elif not inside and start is not None:
            runs.append((start, i))
            start = None
    return runs


def resample_to_grid(t_src: np.ndarray, values: np.ndarray, valid: np.ndarray,
                     rate_hz: float, duration: Optional[float] = None,
                     max_gap: float = 0.12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Irregular samples -> uniform grid at ``rate_hz`` starting at t=0.

    For each grid time take the nearest *valid* sample on either side. If both
    exist and are at most ``max_gap`` seconds apart, interpolate linearly
    (this bridges a single missed detection, ~1-3 source frames); otherwise
    use the nearer one if it is within half a grid period; otherwise the grid
    frame is invalid (NaN). A detection gap longer than ``max_gap`` is never
    interpolated across.

    Returns ``(t_grid, values_grid [N, C], valid_grid [N])``.
    """
    t_src = np.asarray(t_src, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    valid = np.asarray(valid, dtype=bool)
    if duration is None:
        duration = float(t_src[-1]) if len(t_src) else 0.0
    n = int(np.floor(duration * rate_hz + 1e-6)) + 1
    t_grid = np.arange(n) / rate_hz
    out = np.full((n, values.shape[1]), np.nan)
    ok = np.zeros(n, dtype=bool)
    vidx = np.flatnonzero(valid)
    if len(vidx) == 0:
        return t_grid, out, ok
    tv, vv = t_src[vidx], values[vidx]
    tol = 0.5 / rate_hz
    hi = np.searchsorted(tv, t_grid, side="right")  # first valid sample with t > grid time
    lo = hi - 1
    for k in range(n):
        i, j = lo[k], hi[k]
        have_i, have_j = i >= 0, j < len(tv)
        if have_i and have_j and (tv[j] - tv[i]) <= max_gap:
            w = (t_grid[k] - tv[i]) / max(tv[j] - tv[i], 1e-9)
            out[k] = (1 - w) * vv[i] + w * vv[j]
            ok[k] = True
            continue
        best, best_d = -1, tol
        for c, have in ((i, have_i), (j, have_j)):
            if have:
                d = abs(tv[c] - t_grid[k])
                if d <= best_d:
                    best, best_d = c, d
        if best >= 0:
            out[k] = vv[best]
            ok[k] = True
    return t_grid, out, ok


def smooth_runs(values: np.ndarray, valid: np.ndarray, cutoff_hz: float = 8.0, fs: float = 30.0,
                order: int = 2) -> np.ndarray:
    """Zero-phase low-pass per contiguous valid run; NaN outside runs.

    filtfilt needs > 3*(order+1)+1 samples; shorter runs get a centred 3-tap
    moving average instead. Runs are filtered independently so a gap never
    bleeds into real motion.
    """
    from scipy.signal import butter, filtfilt

    values = np.asarray(values, dtype=float)
    squeeze = values.ndim == 1
    if squeeze:
        values = values[:, None]
    valid = np.asarray(valid, dtype=bool)
    out = np.full_like(values, np.nan)
    nyq = 0.5 * fs
    b, a = butter(order, min(max(cutoff_hz / nyq, 1e-3), 0.99), btype="low")
    padlen = 3 * max(len(a), len(b))
    for s, e in contiguous_runs(valid):
        seg = values[s:e]
        if len(seg) > padlen + 1:
            out[s:e] = filtfilt(b, a, seg, axis=0, padtype="odd", padlen=padlen)
        elif len(seg) >= 3:
            k = np.ones(3) / 3.0
            padded = np.pad(seg, ((1, 1), (0, 0)), mode="edge")
            out[s:e] = np.stack([np.convolve(padded[:, c], k, mode="valid") for c in range(seg.shape[1])], axis=1)
        else:
            out[s:e] = seg
    return out[:, 0] if squeeze else out


def hysteresis(x: np.ndarray, on: float, off: float, min_on: int = 1, min_off: int = 1) -> np.ndarray:
    """Two-threshold gate with dwell: switch on after ``min_on`` consecutive
    samples above ``on``; switch off after ``min_off`` consecutive samples
    below ``off``. Returns a bool mask."""
    x = np.asarray(x, dtype=float)
    out = np.zeros(len(x), dtype=bool)
    state, run = False, 0
    for i, v in enumerate(x):
        if not state:
            run = run + 1 if v > on else 0
            if run >= min_on:
                state, run = True, 0
                out[i - min_on + 1:i + 1] = True
        else:
            run = run + 1 if v < off else 0
            if run >= min_off:
                state, run = False, 0
                out[i - min_off + 1:i + 1] = False
        if state:
            out[i] = True
    return out


def segments_to_mask(segments: Sequence[Tuple[float, float]], t_grid: np.ndarray) -> np.ndarray:
    """[(start_s, end_s), ...] -> bool mask over grid times."""
    t_grid = np.asarray(t_grid, dtype=float)
    mask = np.zeros(len(t_grid), dtype=bool)
    for s, e in segments:
        mask |= (t_grid >= s) & (t_grid < e)
    return mask
