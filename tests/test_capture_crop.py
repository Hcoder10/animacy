"""Small-face fallback of ``capture.Trackers`` (YuNet -> crop -> landmarker -> mapped back).

The pure geometry is covered in ``test_capture_math.py``. This file runs the real models
on real frames and is skipped when the models or the reference video are not present.
Two frames of a large-face video are shrunk to a third and pasted into a corner of a
blank canvas, which is exactly the wide-shot failure mode (a 0-detections clip). The
crop path must find the face and its RELATIVE motion (frame B minus frame A) must agree
with the full-frame path: rotation within a few degrees, translation scaled by the
shrink factor (the shrunk face is "three times farther away").
"""
from __future__ import annotations

import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO = os.path.join(ROOT, "data", "raw", "2015_02_07_President_Obama_s_Weekly_Address.webm")
MODELS = [os.path.join(ROOT, "data", "models", f) for f in ("face_landmarker.task", "face_detection_yunet_2023mar.onnx")]

pytestmark = pytest.mark.skipif(not (os.path.exists(VIDEO) and all(os.path.exists(m) for m in MODELS)),
                                reason="needs the reference video in data/raw and the .task/.onnx models")


def _frames(times):
    import cv2

    cap = cv2.VideoCapture(VIDEO)
    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        assert ok
        out.append(fr)
    cap.release()
    return out


def _shrink_into_canvas(frame, scale=0.33, offset=(40, 30)):
    import cv2

    h, w = frame.shape[:2]
    small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.full_like(frame, 96)
    x0, y0 = offset
    canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    return canvas


def test_small_face_is_found_and_relative_motion_matches_full_frame():
    from animacy.capture import Trackers

    frames = _frames([30.0, 60.0, 90.0])
    scale = 0.33

    full = Trackers(want_pose=False, allow_crop=False)
    ref = [full.detect(f, i / 30.0) for i, f in enumerate(frames)]
    assert all(r["face_ok"] and r["face_crop"] is None for r in ref)

    no_crop = Trackers(want_pose=False, allow_crop=False)
    composites = [_shrink_into_canvas(f, scale) for f in frames]
    assert not any(no_crop.detect(c, i / 30.0)["face_ok"] for i, c in enumerate(composites)), "shrunk face should defeat the full-frame detector"

    crop = Trackers(want_pose=False, allow_crop=True)
    got = [crop.detect(c, i / 30.0) for i, c in enumerate(composites)]
    assert all(g["face_ok"] and g["face_crop"] is not None for g in got)
    assert crop.crop_mode and crop.crop_frames == 3

    # landmarks map back into the pasted region of the canvas
    lm = got[0]["lm_xy"] * np.array([composites[0].shape[1], composites[0].shape[0]])
    assert 40 <= lm[:, 0].min() and lm[:, 0].max() <= 40 + 854 * scale + 2
    assert 30 <= lm[:, 1].min() and lm[:, 1].max() <= 30 + 480 * scale + 2

    # relative rotation between frames agrees with the full-frame path
    for a, b in ((0, 1), (1, 2)):
        d_ref = np.array(ref[b]["head_angles"]) - np.array(ref[a]["head_angles"])
        d_got = np.array(got[b]["head_angles"]) - np.array(got[a]["head_angles"])
        assert np.all(np.abs(d_ref - d_got) < 4.0), (d_ref, d_got)
    # relative translation: the same physical head motion seen from 1/scale farther away keeps
    # its LATERAL displacement in mm (pixel motion shrinks by scale, depth grows by 1/scale, the
    # two cancel) while DEPTH changes (from face-size changes) scale with the distance, 1/scale.
    for a, b in ((0, 1), (1, 2)):
        d_ref = np.array(ref[b]["head_trans"]) - np.array(ref[a]["head_trans"])  # (x=depth, y, z) mm
        d_got = np.array(got[b]["head_trans"]) - np.array(got[a]["head_trans"])
        expect = np.array([1 / scale, 1.0, 1.0])
        big = np.abs(d_ref) > 8.0  # only judge components that actually moved (> 8 mm)
        if big.any():
            ratio = d_got[big] / d_ref[big]
            assert np.all(ratio > 0), (d_ref, d_got)
            assert np.all(np.abs(ratio / expect[big] - 1.0) < 0.4), (d_ref, d_got, ratio)
    # absolute depth: the shrunk face reads ~1/scale farther from the camera
    z_ref = np.mean([r["head_trans"][0] for r in ref])
    z_got = np.mean([g["head_trans"][0] for g in got])
    assert 0.7 / scale < z_got / z_ref < 1.3 / scale, (z_ref, z_got)
