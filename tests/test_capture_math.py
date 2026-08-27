"""Pure-math tests for the capture stage: no camera, no MediaPipe models."""
from __future__ import annotations

import numpy as np
import pytest

from animacy import capture_math as cm
from animacy.schema import ARM_CHANNELS, CHANNELS, FACE_CHANNELS, HumanClip, empty_frames


def cam_matrix(r_cam: np.ndarray, t_cam_cm=(0.0, 0.0, -40.0)) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = r_cam
    m[:3, 3] = t_cam_cm
    return m


def angles(m):
    r_body, _ = cm.head_pose_from_matrix(m)
    return cm.head_angles_deg(r_body)


# ---------------------------------------------------------------- frames / rotations
def test_cam_to_body_is_proper_rotation():
    p = cm.CAM_TO_BODY
    assert np.allclose(p @ p.T, np.eye(3))
    assert np.isclose(np.linalg.det(p), 1.0)
    # cam +Z (toward viewer) -> body +x (forward); cam +X (image right) -> body +y (subject's left); cam +Y -> up
    assert np.allclose(p @ [0, 0, 1], [1, 0, 0])
    assert np.allclose(p @ [1, 0, 0], [0, 1, 0])
    assert np.allclose(p @ [0, 1, 0], [0, 0, 1])


def test_identity_is_zero_angles():
    assert np.allclose(angles(cam_matrix(np.eye(3))), (0, 0, 0), atol=1e-9)


def test_turn_left_is_positive_yaw():
    # In the camera frame the nose points along +Z; turning to the subject's LEFT swings it
    # toward +X (image right). Ry(+20) maps Z -> (sin20, 0, cos20): that is a left turn.
    yaw, pitch, roll = angles(cam_matrix(cm.rotation_about([0, 1, 0], 20)))
    assert yaw == pytest.approx(20, abs=1e-6)
    assert abs(pitch) < 1e-6 and abs(roll) < 1e-6


def test_look_up_is_positive_pitch():
    # Looking up tilts the nose (+Z) toward +Y. Rx(phi) maps Z -> (0, -sin phi, cos phi), so up = Rx(-20).
    yaw, pitch, roll = angles(cam_matrix(cm.rotation_about([1, 0, 0], -20)))
    assert pitch == pytest.approx(20, abs=1e-6)
    assert abs(yaw) < 1e-6 and abs(roll) < 1e-6


def test_right_ear_down_is_positive_roll():
    # Right ear toward the right shoulder tilts the head's up vector (+Y) toward the subject's
    # right (-X). Rz(psi) maps Y -> (-sin psi, cos psi, 0): psi = +20 is right-ear-down.
    yaw, pitch, roll = angles(cam_matrix(cm.rotation_about([0, 0, 1], 20)))
    assert roll == pytest.approx(20, abs=1e-6)
    assert abs(yaw) < 1e-6 and abs(pitch) < 1e-6


def test_translation_mapping_cm_to_mm():
    _, t = cm.head_pose_from_matrix(cam_matrix(np.eye(3), (1.0, 2.0, -3.0)))
    # cam (x right, y up, z toward camera) -> body (x fwd, y left, z up), cm -> mm
    assert np.allclose(t, [-30.0, 10.0, 20.0])


def test_euler_roundtrip_random():
    rng = np.random.default_rng(0)
    for _ in range(50):
        y, p, r = rng.uniform(-80, 80), rng.uniform(-60, 60), rng.uniform(-45, 45)
        rm = cm.head_rotmat_from_angles_deg(y, p, r)
        assert np.allclose(cm.head_angles_deg(rm), (y, p, r), atol=1e-6)


def test_neutral_zeroing():
    neutral = {"head_angles_deg": [5.0, 0.0, 0.0], "head_trans_mm": [-400.0, 50.0, 60.0]}
    r = cm.head_rotmat_from_angles_deg(15.0, 0.0, 0.0)
    yaw, pitch, roll, x, y, z = cm.relative_head(r, np.array([-390.0, 40.0, 65.0]), neutral)
    assert yaw == pytest.approx(10.0, abs=1e-6)
    assert abs(pitch) < 1e-6 and abs(roll) < 1e-6
    assert (x, y, z) == pytest.approx((10.0, -10.0, 5.0))
    # the neutral pose itself maps to all-zeros
    r_n = cm.head_rotmat_from_angles_deg(*neutral["head_angles_deg"])
    assert np.allclose(cm.relative_head(r_n, neutral["head_trans_mm"], neutral), 0.0, atol=1e-9)


def test_neutral_pose_is_median():
    ang = np.array([[1, 2, 3], [10, 20, 30], [2, 3, 4]], float)
    tr = np.array([[0, 0, 0], [100, 100, 100], [1, 1, 1]], float)
    n = cm.neutral_pose(ang, tr)
    assert n["head_angles_deg"] == [2, 3, 4]
    assert n["head_trans_mm"] == [1, 1, 1]


# ---------------------------------------------------------------- blendshapes
def test_gaze_signs_from_blendshapes():
    raw = cm.face_raw_from_blendshapes({"eyeLookOutLeft": 1.0, "eyeLookInRight": 1.0})
    assert raw["gaze_yaw"] > 0  # both eyes toward the subject's left
    raw = cm.face_raw_from_blendshapes({"eyeLookUpLeft": 1.0, "eyeLookUpRight": 1.0})
    assert raw["gaze_pitch"] > 0
    raw = cm.face_raw_from_blendshapes({"eyeLookInLeft": 1.0, "eyeLookOutRight": 1.0})
    assert raw["gaze_yaw"] < 0


def test_brow_neutral_zeroing():
    neutral_raw = cm.face_raw_from_blendshapes({"browDownLeft": 0.6, "browDownRight": 0.6})
    rest = cm.face_channels_relative(neutral_raw, neutral_raw)
    assert rest["brow_l"] == 0 and rest["brow_r"] == 0 and rest["brow_furrow"] == 0
    raised = cm.face_channels_relative(cm.face_raw_from_blendshapes({"browInnerUp": 0.8, "browOuterUpLeft": 0.8, "browOuterUpRight": 0.8}), neutral_raw)
    assert raised["brow_l"] > 0.9 and raised["brow_r"] > 0.9 and raised["brow_furrow"] == 0
    furrow = cm.face_channels_relative(cm.face_raw_from_blendshapes({"browDownLeft": 1.0, "browDownRight": 1.0}), neutral_raw)
    assert furrow["brow_furrow"] == pytest.approx(0.4) and furrow["brow_l"] == 0
    assert cm.face_channels_relative(cm.face_raw_from_blendshapes({"eyeBlinkLeft": 1.0, "jawOpen": 0.5}), neutral_raw)["eye_open_l"] == 0
    assert cm.face_channels_relative(cm.face_raw_from_blendshapes({"jawOpen": 0.5}), neutral_raw)["mouth_open"] == 0.5


def test_iris_gaze_cue_sign():
    lm = np.zeros((478, 2))
    lm[362], lm[263], lm[133], lm[33] = [0.6, 0.5], [0.7, 0.5], [0.4, 0.5], [0.3, 0.5]
    lm[473], lm[468] = [0.68, 0.5], [0.38, 0.5]  # both irises toward image right
    assert cm.iris_gaze_yaw_unit(lm) > 0.5


# ---------------------------------------------------------------- pose
def test_pose_to_body_axes():
    assert np.allclose(cm.pose_to_body(np.array([[1.0, 0, 0]])), [[0, 1, 0]])   # image right -> subject's left
    assert np.allclose(cm.pose_to_body(np.array([[0, 1.0, 0]])), [[0, 0, -1]])  # image down -> -up
    assert np.allclose(cm.pose_to_body(np.array([[0, 0, 1.0]])), [[-1, 0, 0]])  # away from camera -> -forward


def _body_pose(**over):
    p = np.zeros((33, 3))
    P = cm.POSE
    p[P["l_shoulder"]] = [0, 0.2, 0.5]
    p[P["r_shoulder"]] = [0, -0.2, 0.5]
    p[P["l_hip"]] = [0, 0.15, 0]
    p[P["r_hip"]] = [0, -0.15, 0]
    for k, v in over.items():
        p[P[k]] = v
    return p


def test_torso_signs():
    t = cm.torso_channels(_body_pose())
    assert all(abs(t[k]) < 1e-9 for k in t)
    lean = cm.torso_channels(_body_pose(l_shoulder=[0.2, 0.2, 0.5], r_shoulder=[0.2, -0.2, 0.5]))
    assert lean["torso_lean_fwd"] > 15 and abs(lean["torso_lean_side"]) < 1e-9
    side = cm.torso_channels(_body_pose(l_shoulder=[0, 0.4, 0.5], r_shoulder=[0, 0.0, 0.5]))
    assert side["torso_lean_side"] > 15
    # turn left: left shoulder goes back (-x), right shoulder comes forward (+x)
    turned = cm.torso_channels(_body_pose(l_shoulder=[-0.1, 0.17, 0.5], r_shoulder=[0.1, -0.17, 0.5]))
    assert turned["torso_yaw"] > 20


def _arm(shoulder, elbow, wrist, index, pinky, thumb, side="right"):
    pre = "l_" if side == "left" else "r_"
    return _body_pose(**{pre + "shoulder": shoulder, pre + "elbow": elbow, pre + "wrist": wrist,
                         pre + "index": index, pre + "pinky": pinky, pre + "thumb": thumb})


def test_arm_hanging_then_forward():
    sh = [0, -0.2, 0.5]
    hang = cm.arm_channels(_arm(sh, [0, -0.2, 0.2], [0, -0.2, -0.05], [0, -0.2, -0.22], [0, -0.23, -0.2], [0.03, -0.18, -0.15]))
    assert hang["shoulder_pitch"] == pytest.approx(0, abs=1e-6)
    assert hang["shoulder_yaw"] == pytest.approx(0, abs=1e-6)
    assert hang["elbow_flex"] == pytest.approx(0, abs=1e-6)
    # straight forward, hand in line with the forearm, thumb straight up (handshake orientation)
    fwd = cm.arm_channels(_arm(sh, [0.3, -0.2, 0.5], [0.55, -0.2, 0.5], [0.72, -0.2, 0.5], [0.70, -0.24, 0.5], [0.62, -0.22, 0.6]))
    assert fwd["shoulder_pitch"] == pytest.approx(90, abs=1e-6)
    assert fwd["shoulder_yaw"] == pytest.approx(0, abs=1e-6)
    assert fwd["elbow_flex"] == pytest.approx(0, abs=1e-6)
    assert fwd["wrist_roll"] == pytest.approx(0, abs=1e-6)  # thumb up
    assert fwd["hand_open"] > 0.5


# upper arm hanging, forearm horizontal forward (elbow 90), hand tipped up, thumb pointing left (pronated)
_TRAY = dict(elbow=[0, -0.2, 0.2], wrist=[0.25, -0.2, 0.2], index=[0.4, -0.2, 0.3], pinky=[0.4, -0.24, 0.3], thumb=[0.35, -0.10, 0.30])


def test_arm_signs_and_mirroring():
    sh = [0, -0.2, 0.5]
    bent = cm.arm_channels(_arm(sh, _TRAY["elbow"], _TRAY["wrist"], _TRAY["index"], _TRAY["pinky"], _TRAY["thumb"]))
    assert bent["elbow_flex"] == pytest.approx(90, abs=1e-6)
    assert bent["wrist_pitch"] > 20         # hand tips up relative to the forearm
    assert bent["wrist_roll"] > 60          # thumb toward the subject's left
    swing = cm.arm_channels(_arm(sh, [0.2, 0.05, 0.5], [0.45, 0.05, 0.5], [0.62, 0.05, 0.5], [0.60, 0.01, 0.5], [0.52, 0.09, 0.55]))
    assert swing["shoulder_yaw"] > 20       # arm swung toward the subject's left
    # the mirrored left arm in the mirrored pose must give identical channels
    left = bent
    mirrored_pose = cm.mirror_left_right(_arm(sh, _TRAY["elbow"], _TRAY["wrist"], _TRAY["index"], _TRAY["pinky"], _TRAY["thumb"]))
    # after mirroring the right-arm landmarks sit where a left arm would; copy them into the left slots
    p = _body_pose()
    for k in ("shoulder", "elbow", "wrist", "index", "pinky", "thumb"):
        p[cm.POSE["l_" + k]] = mirrored_pose[cm.POSE["r_" + k]]
    left_from_left = cm.arm_channels(p, side="left")
    for k in ARM_CHANNELS:
        assert left_from_left[k] == pytest.approx(left[k], abs=1e-6), k


# ---------------------------------------------------------------- time series
def test_resample_25_to_30_linear():
    t = np.arange(0, 2.0, 1 / 25)
    v = np.stack([t, 2 * t], axis=1)
    tg, vg, ok = cm.resample_to_grid(t, v, np.ones(len(t), bool), 30.0)
    assert np.allclose(np.diff(tg), 1 / 30)
    assert ok.all()
    assert np.allclose(vg[:, 0], tg, atol=1e-9) and np.allclose(vg[:, 1], 2 * tg, atol=1e-9)


def test_resample_60_to_30_and_gap():
    t = np.arange(0, 3.0, 1 / 60)
    valid = np.ones(len(t), bool)
    valid[(t > 1.0) & (t < 1.5)] = False
    tg, vg, ok = cm.resample_to_grid(t, np.sin(t), valid, 30.0)
    assert np.allclose(vg[ok, 0], np.sin(tg[ok]), atol=1e-3)
    gap = (tg > 1.0 + 1 / 30) & (tg < 1.5 - 1 / 30)
    assert not ok[gap].any() and np.isnan(vg[gap, 0]).all()
    assert ok[tg < 0.9].all() and ok[tg > 1.6].all()


def test_resample_never_bridges_gap():
    t = np.array([0.0, 0.1, 1.0, 1.1])  # 0.9 s hole with valid samples either side
    tg, vg, ok = cm.resample_to_grid(t, np.array([0.0, 1.0, 10.0, 11.0]), np.ones(4, bool), 30.0)
    mid = (tg > 0.15) & (tg < 0.95)
    assert not ok[mid].any()


def test_resample_bridges_single_dropout_only():
    t = np.arange(0, 1.0, 1 / 30)
    valid = np.ones(len(t), bool)
    valid[10] = False            # one missed detection: neighbours 67 ms apart -> bridged
    valid[20:25] = False         # five missed: neighbours 200 ms apart -> hole
    tg, vg, ok = cm.resample_to_grid(t, t, valid, 30.0)
    assert ok[10] and vg[10, 0] == pytest.approx(t[10], abs=1e-9)
    assert not ok[21:24].any()


def test_smooth_runs_reduces_noise_and_respects_gaps():
    rng = np.random.default_rng(1)
    t = np.arange(300) / 30.0
    clean = np.sin(2 * np.pi * 0.5 * t)
    noisy = clean + rng.normal(0, 0.1, len(t))
    valid = np.ones(len(t), bool)
    valid[120:150] = False
    noisy[~valid] = np.nan
    sm = cm.smooth_runs(noisy, valid, cutoff_hz=8.0, fs=30.0)
    assert np.isnan(sm[~valid]).all()
    err_before = np.abs(noisy[valid] - clean[valid]).mean()
    err_after = np.abs(sm[valid] - clean[valid]).mean()
    assert err_after < err_before * 0.8
    # short run (5 frames) does not crash and stays finite
    short_valid = np.zeros(20, bool)
    short_valid[3:8] = True
    out = cm.smooth_runs(np.arange(20.0), short_valid)
    assert np.isfinite(out[3:8]).all() and np.isnan(out[:3]).all()


def test_hysteresis_dwell():
    x = np.zeros(40)
    x[5] = 10          # one-sample spike: must not trigger with min_on=3
    x[10:25] = 10      # sustained: triggers
    x[15] = 0          # single dip: must not switch off with min_off=3
    m = cm.hysteresis(x, on=5, off=2, min_on=3, min_off=3)
    assert not m[5]
    assert m[12:25].all()
    assert m[15]
    assert not m[30:].any()
    assert m[10:13].all()  # the dwell frames themselves are marked


def test_segments_to_mask():
    tg = np.arange(0, 1.0, 0.1)
    m = cm.segments_to_mask([(0.2, 0.45)], tg)
    assert m.sum() == 3 and m[2] and m[4] and not m[5]


# ---------------------------------------------------------------- writer
def test_empty_frames_writer_roundtrip(tmp_path):
    clip = HumanClip.from_frames(empty_frames(90), source="test", rate_hz=30.0)
    assert clip.validate() == []
    clip.audio = np.zeros(int(3.0 * 16000), np.float32)
    clip.save(str(tmp_path / "clip"))
    back = HumanClip.load(str(tmp_path / "clip"))
    assert back.validate() == [] and len(back) == 90 and list(back.frames.columns) == CHANNELS


def test_build_frames_synthetic_passes_validate(tmp_path):
    """The capture assembler on synthetic samples (no models): valid clip, NaN where invalid."""
    from animacy.capture import build_frames

    rng = np.random.default_rng(2)
    samples = []
    for i in range(150):  # 5 s at 30 Hz
        t = i / 30.0
        face_ok = not (60 <= i < 75)
        s = {"t": t, "face_ok": face_ok, "pose_ok": face_ok, "arm_ok": i % 2 == 0, "frame_idx": i}
        if face_ok:
            s["head_angles"] = (5 + 10 * np.sin(t), -2 + 5 * np.cos(t), 1.0)
            s["head_trans"] = np.array([-400 + 20 * np.sin(t), 50, 60])
            s["face_raw"] = cm.face_raw_from_blendshapes({"jawOpen": float(rng.uniform(0, 0.5)), "browDownLeft": 0.5, "browDownRight": 0.5})
            s["torso_vals"] = {"torso_lean_fwd": 20 + np.sin(t), "torso_lean_side": 1.0, "torso_yaw": -3.0}
        if s["arm_ok"]:
            s["arm_vals"] = {"shoulder_yaw": 10.0, "shoulder_pitch": 80.0, "elbow_flex": 30.0, "wrist_roll": 5.0, "wrist_pitch": -10.0, "hand_open": 0.7}
        samples.append(s)
    frames, extra = build_frames(samples, neutral_seconds=1.0)
    clip = HumanClip.from_frames(frames, source="synthetic", rate_hz=30.0)
    assert clip.validate() == []
    assert len(frames) == 150
    fv = frames["face_valid"].to_numpy()
    assert fv[:55].all() and not fv[62:73].any()
    assert np.isnan(frames.loc[65, FACE_CHANNELS].to_numpy(dtype=float)).all()
    assert frames["arm_valid"].mean() > 0.9  # single missed detections (valid neighbours 67 ms apart) are bridged
    # neutral window = first second of valid frames; head_yaw there is ~0 after zeroing
    assert abs(frames["head_yaw"].iloc[:20].mean()) < 3.0
    assert extra["stats"]["face_valid_frac"] == pytest.approx(fv.mean())
    clip.save(str(tmp_path / "syn"))
    assert HumanClip.load(str(tmp_path / "syn")).validate() == []


# ---------------------------------------------------------------- small-face crop geometry
def test_crop_box_is_square_clamped_and_min_sized():
    assert cm.crop_box_for_face(400, 240, 60, 854, 480, margin=2.5, min_side=192) == (304, 144, 192)  # 60*2.5=150 < 192
    x0, y0, side = cm.crop_box_for_face(20, 10, 100, 854, 480, margin=2.5, min_side=192)
    assert (x0, y0, side) == (0, 0, 250)  # clamped into the image
    x0, y0, side = cm.crop_box_for_face(850, 470, 300, 854, 480)
    assert side == 480 and x0 + side <= 854 and y0 + side <= 480


def test_crop_landmarks_to_full():
    lm = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.25]])
    out = cm.crop_landmarks_to_full(lm, (100, 50, 200), (800, 400))
    assert np.allclose(out, [[100 / 800, 50 / 400], [300 / 800, 250 / 400], [200 / 800, 100 / 400]])


def test_crop_to_full_translation_identity_and_scaling():
    # a crop that IS the full (square) frame changes nothing
    t = np.array([3.0, -2.0, -40.0])
    assert np.allclose(cm.crop_to_full_translation(t, (0, 0, 480), (480, 480)), t)
    # a centred crop of a third of the height: depth triples, lateral position unchanged
    out = cm.crop_to_full_translation(t, (267, 160, 160), (854, 480))  # centre (347, 240) vs image centre (427, 240)
    assert out[2] == pytest.approx(-120.0)
    # crop centre 80 px left of the image centre -> x shifts left by 80 * |z| / f_crop
    f_crop = 80 / np.tan(np.radians(31.5))
    assert out[0] == pytest.approx(3.0 - 80 * 40.0 / f_crop)
    assert out[1] == pytest.approx(-2.0)
    # crop centre below the image centre -> y (up) decreases
    out = cm.crop_to_full_translation(t, (347, 200, 160), (854, 480))  # centre (427, 280): 40 px below
    assert out[1] == pytest.approx(-2.0 - 40 * 40.0 / f_crop) and out[0] == pytest.approx(3.0)


# ---------------------------------------------------------------- --pose-every N
def _synthetic_samples(n=150, pose_every=1, dropout=None):
    """30 fps samples: face on every frame, pose only on every ``pose_every``-th frame
    (``dropout`` = (a, b) range of frames with no pose at all, a real tracking hole)."""
    samples = []
    for i in range(n):
        t = i / 30.0
        s = {"t": t, "face_ok": True, "pose_ok": False, "arm_ok": False, "frame_idx": i,
             "head_angles": (10 * np.sin(2 * np.pi * 0.4 * t), 3 * np.cos(2 * np.pi * 0.3 * t), 1.0),
             "head_trans": np.array([-400 + 15 * np.sin(2 * np.pi * 0.5 * t), 50.0, 60.0]),
             "face_raw": cm.face_raw_from_blendshapes({"jawOpen": 0.3 + 0.2 * np.sin(2 * np.pi * 2 * t)})}
        has_pose = (i % pose_every == 0) and not (dropout and dropout[0] <= i < dropout[1])
        s["pose_skipped"] = i % pose_every != 0
        if has_pose:
            s["pose_ok"] = s["arm_ok"] = True
            s["torso_vals"] = {"torso_lean_fwd": 20 + 4 * np.sin(2 * np.pi * 0.5 * t), "torso_lean_side": 1.0,
                               "torso_yaw": -3 + 5 * np.cos(2 * np.pi * 0.4 * t)}
            s["arm_vals"] = {"shoulder_yaw": 10 + 8 * np.sin(2 * np.pi * 0.6 * t), "shoulder_pitch": 80.0, "elbow_flex": 30.0,
                             "wrist_roll": 5.0, "wrist_pitch": -10.0, "hand_open": 0.5 + 0.3 * np.sin(2 * np.pi * 0.5 * t)}
        samples.append(s)
    return samples


@pytest.mark.parametrize("n_every", [2, 3, 4])
def test_pose_every_keeps_face_and_interpolates_torso(n_every):
    from animacy.capture import build_frames
    from animacy.schema import ARM_CHANNELS, FACE_CHANNELS, TORSO_CHANNELS, HumanClip

    full, _ = build_frames(_synthetic_samples(pose_every=1), neutral_seconds=0.0, pose_every=1)
    dec, extra = build_frames(_synthetic_samples(pose_every=n_every), neutral_seconds=0.0, pose_every=n_every)
    assert HumanClip.from_frames(dec, source="synthetic").validate() == []
    # face channels: identical (the face never skipped a frame)
    for c in FACE_CHANNELS + ["face_valid"]:
        assert np.allclose(full[c].to_numpy(), dec[c].to_numpy(), atol=1e-5, equal_nan=True), c
    # torso/arm: every grid frame bridged (real detections N frames apart, not dropouts) ...
    assert not dec[TORSO_CHANNELS].isna().any().any()
    assert dec["arm_valid"].to_numpy().all()
    # ... and linear interpolation of a slow sinusoid lands within tolerance of the every-frame result
    # in the interior; the last frames past the final pose sample are HELD (edge rule), so they may
    # lag by up to one frame of motion (a 4 deg / 0.5 Hz sinusoid moves 0.42 deg per frame)
    # (torso channels are neutral-zeroed by the median over the pose samples, so a 1-in-N subsample
    # shifts them by a small constant; that offset is bounded separately and removed before comparing)
    for c in TORSO_CHANNELS + ARM_CHANNELS:
        d = full[c].to_numpy() - dec[c].to_numpy()
        offset = float(np.nanmedian(d)) if c in TORSO_CHANNELS else 0.0
        assert abs(offset) < 0.6, (c, offset)
        diff = np.abs(d - offset)
        interior, edge = np.nanmax(diff[: -n_every - 1]), np.nanmax(diff[-n_every - 1:])
        assert interior < (0.03 if c == "hand_open" else 0.35), (c, interior)
        # held edge: at most N-1 frames of lag; the fastest synthetic signal moves ~1 deg (0.03 hand_open) per frame
        assert edge < (0.05 if c == "hand_open" else 1.1) * n_every, (c, edge)
    assert extra["stats"]["torso_valid_frac"] == 1.0


def test_pose_every_does_not_bridge_a_real_hole():
    from animacy.capture import build_frames

    dec, _ = build_frames(_synthetic_samples(pose_every=2, dropout=(60, 90)), neutral_seconds=0.0, pose_every=2)
    tv = ~dec["torso_lean_fwd"].isna().to_numpy()
    assert tv[:58].all() and tv[92:].all()
    assert not tv[64:86].any()  # a 1 s hole stays NaN: the widened gap (0.12 s for N=2) does not cover it
