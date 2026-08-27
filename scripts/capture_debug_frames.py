"""Verify capture sign conventions on real video; dump annotated frames.

Two independent checks, because a sign must not be guessed:

1. **Image-space cross-check** (automatic). For every face-valid frame the
   normalized face-mesh landmarks give convention-free cues:
     yaw_cue   = nose x offset from the cheek midpoint (+ = image right = subject's left)
     roll_cue  = right-eye-outer y minus left-eye-outer y (+ = right eye lower = right ear drops)
     pitch_cue = nose height above the cheek/ear midpoint (+ = nose up = looking up)
     size_cue  = cheek-to-cheek width (+ = closer to the camera)
     cx_cue    = face centre x (+ = image right = subject's left)
     cy_cue    = -face centre y (+ = up)
     iris_cue  = iris offset within the eye (+ = image right = subject's left)
   Each is correlated with the canonical channel it should track; a positive
   correlation confirms the sign, a negative one would mean the mapping is
   mirrored.
2. **Annotated frames** (for a human/agent to read). For each channel the
   frames at the 3rd and 97th percentile are written to ``data/debug/<stem>/``
   with the channel values drawn on them, plus the landmark points used for
   the cues (yellow) and the eye line, so what the picture shows can be
   compared with what the number claims.

Usage:
    python scripts/capture_debug_frames.py data/raw/talk.webm [--duration 120] [--out data/debug]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from animacy import capture as cap  # noqa: E402
from animacy import capture_math as cm  # noqa: E402
from animacy.schema import TORSO_CHANNELS  # noqa: E402

EXPECT = {
    "head_yaw": "subject turned to THEIR LEFT (nose toward image RIGHT)",
    "head_pitch": "subject looking UP (chin up)",
    "head_roll": "subject's RIGHT ear dropped toward their right shoulder (head tilted toward image LEFT)",
    "head_x": "subject CLOSER to the camera (face larger)",
    "head_y": "subject shifted to THEIR LEFT (image RIGHT)",
    "head_z": "subject's head HIGHER in the frame",
    "gaze_yaw": "eyes looking to the subject's LEFT (irises toward image RIGHT)",
    "torso_lean_fwd": "shoulders leaning toward the camera",
    "torso_yaw": "shoulders turned to the subject's LEFT",
    "shoulder_pitch": "upper arm raised (0 hanging, 90 horizontal, 180 up)",
    "shoulder_yaw": "upper arm swung toward the subject's LEFT (across the body for the right arm)",
    "elbow_flex": "elbow bent (0 straight)",
    "wrist_roll": "thumb rotated toward the subject's LEFT (palm down for the right arm)",
    "hand_open": "fingers spread / extended (0 fist)",
}
CUES = {  # channel -> (cue name, expected correlation sign)
    "head_yaw": ("yaw_cue", +1), "head_roll": ("roll_cue", +1), "head_pitch": ("pitch_cue", +1),
    "head_x": ("size_cue", +1), "head_y": ("cx_cue", +1), "head_z": ("cy_cue", +1), "gaze_yaw": ("iris_cue", +1),
}


def cues_from_landmarks(lm: np.ndarray, aspect: float) -> dict:
    """Convention-free image-space cues (x normalized by width, y by height; y is down)."""
    xs = lm[:, 0]
    ys = lm[:, 1] / aspect  # make y units comparable to x
    width = xs[454] - xs[234]  # subject's left cheek (image right) minus right cheek
    return {
        "yaw_cue": float((xs[1] - 0.5 * (xs[234] + xs[454])) / max(width, 1e-6)),
        "roll_cue": float((ys[33] - ys[263]) / max(xs[263] - xs[33], 1e-6)),
        "pitch_cue": float(-(ys[1] - 0.5 * (ys[234] + ys[454])) / max(width, 1e-6)),
        "size_cue": float(width),
        "cx_cue": float(0.5 * (xs[234] + xs[454])),
        "cy_cue": float(-0.5 * (ys[234] + ys[454])),
        "iris_cue": float(cm.iris_gaze_yaw_unit(lm)),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--duration", type=float, default=0.0)
    p.add_argument("--out", default=os.path.join("data", "debug"))
    p.add_argument("--neutral-seconds", type=float, default=0.0)
    a = p.parse_args()

    import cv2

    stem = os.path.splitext(os.path.basename(a.video))[0][:40]
    out_dir = os.path.join(a.out, stem)
    os.makedirs(out_dir, exist_ok=True)

    samples, _, info = cap.run_source(a.video, "right", a.duration, False, want_audio=False)
    aspect = info["width"] / max(info["height"], 1)
    frames, extra = cap.build_frames(samples, a.neutral_seconds)
    neutral = extra["neutral"]

    # per-sample relative channels + cues (same math as build_frames, before resampling)
    rel_rows, cue_rows, ok_idx = [], [], []
    for i, s in enumerate(samples):
        if not s["face_ok"]:
            continue
        r_body = cm.head_rotmat_from_angles_deg(*s["head_angles"])
        yaw, pitch, roll, x, y, z = cm.relative_head(r_body, s["head_trans"], neutral)
        fv = cm.face_channels_relative(s["face_raw"], neutral["face_raw"])
        rel = {"head_yaw": yaw, "head_pitch": pitch, "head_roll": roll, "head_x": x, "head_y": y, "head_z": z, **fv}
        if s.get("pose_ok"):
            for c, n in zip(TORSO_CHANNELS, neutral["torso_deg"]):
                rel[c] = s["torso_vals"][c] - n
        if s.get("arm_ok"):
            rel.update(s["arm_vals"])
        s["rel"] = rel
        rel_rows.append(rel)
        cue_rows.append(cues_from_landmarks(s["lm_xy"], aspect))
        ok_idx.append(i)
    if len(ok_idx) < 30:
        print(f"only {len(ok_idx)} face-valid samples; not enough to verify")
        return 1
    print(f"{a.video}: {len(samples)} samples, {len(ok_idx)} face-valid ({len(ok_idx) / len(samples):.0%})")

    print("\nimage-space cross-check (Pearson r between canonical channel and convention-free cue):")
    verdicts = {}
    for ch, (cue, sign) in CUES.items():
        v = np.array([r[ch] for r in rel_rows])
        c = np.array([q[cue] for q in cue_rows])
        m = ~(np.isnan(v) | np.isnan(c))
        r = float(np.corrcoef(v[m], c[m])[0, 1]) if m.sum() > 10 and v[m].std() > 1e-6 and c[m].std() > 1e-6 else float("nan")
        ok = (r * sign) > 0.3
        verdicts[ch] = (r, ok)
        print(f"  {ch:12s} vs {cue:10s} r={r:+.3f}  std={v[m].std():6.2f}  {'PASS' if ok else 'WEAK/FAIL'}")

    # annotated frames at the 3rd / 97th percentile of each channel
    wanted = {}
    for ch in list(EXPECT):
        vals = np.array([r.get(ch, np.nan) for r in rel_rows])
        m = ~np.isnan(vals)
        if m.sum() < 10:
            continue
        for tag, q in (("min", 3), ("max", 97)):
            target = np.percentile(vals[m], q)
            k = int(np.argmin(np.where(m, np.abs(vals - target), np.inf)))
            si = ok_idx[k]
            wanted.setdefault(samples[si]["frame_idx"], []).append((ch, tag, float(vals[k]), si))

    capv = cv2.VideoCapture(a.video)
    idx, written = 0, []
    while True:
        ok, frame = capv.read()
        if not ok:
            break
        if idx in wanted:
            for ch, tag, val, si in wanted[idx]:
                s = samples[si]
                title = f"{ch} {tag.upper()} = {val:+.1f}  (frame {idx}, t={s['t']:.2f}s)"
                img = cap.draw_overlay(frame, s, rel=s["rel"], title=title)
                note = f"if {ch}>0 expect: {EXPECT[ch]}"
                cv2.putText(img, note, (6, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, note, (6, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                path = os.path.join(out_dir, f"{ch}_{tag}.jpg")
                cv2.imwrite(path, img)
                written.append(path)
        idx += 1
    capv.release()
    print(f"\nwrote {len(written)} annotated frames to {out_dir}")
    for w in written:
        print("  ", w)
    return 0 if all(v[1] for v in verdicts.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
