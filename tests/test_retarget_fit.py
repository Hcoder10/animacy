"""animacy.retarget_fit: statistics, gain proposals, gaze compensation, ROBOT.md surgery."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from animacy.profile import Profile
from animacy.retarget_fit import (
    JointStats, gaze_compensation, joint_stats, library_center, propose_multipliers, rewrite_gains, velocity_w1,
)


def _table(vals, rate=30.0):
    vals = np.asarray(vals, dtype=float)
    return pd.DataFrame({"t": np.arange(len(vals)) / rate, "a": vals})


def test_joint_stats_percentiles_velocity_and_stillness():
    v = np.concatenate([np.zeros(50), np.linspace(0, 30, 31), np.full(19, 30.0)])
    st = joint_stats([_table(v)], ["a"], {"a": 0.0})["a"]
    assert st.n == 100 and abs(st.p50 - 0.0) < 1e-9 and abs(st.abs_p99 - 30.0) < 1e-9
    assert abs(st.vel_max - 30.0) < 1e-9          # 1 unit per frame at 30 Hz
    assert 0.6 < st.still < 0.75                  # 68 of 99 steps are 0
    assert library_center([_table(v + 5)], ["a"])["a"] == pytest.approx(5.0)
    assert velocity_w1(st, st) == 0.0


def _stats(values, speeds):
    values = np.asarray(values, float)
    speeds = np.asarray(speeds, float)
    return JointStats(len(values), float(np.percentile(values, 5)), float(np.percentile(values, 50)), float(np.percentile(values, 95)),
                      float(np.percentile(np.abs(values), 95)), float(np.percentile(np.abs(values), 99)),
                      float(np.percentile(speeds, 50)), float(np.percentile(speeds, 95)), float(np.percentile(speeds, 99)), float(speeds.max()),
                      float(np.mean(speeds < 5)), speeds, values)


def _prof(lo=-90, hi=90, rest=0.0, mmin=None, mmax=None):
    m = {"from": "head_yaw", "gain": 1.0}
    if mmin is not None:
        m["min"] = mmin
    if mmax is not None:
        m["max"] = mmax
    return Profile(**{"schema": "animacy.robot.v1", "name": "t", "display_name": "T", "description": {"urdf": "x.urdf"},
                      "joints": [{"name": "a", "min": lo, "max": hi, "rest": rest, "max_speed": 100}],
                      "retarget": {"default": {"a": m}}})


def test_propose_multipliers_fit_caps_headroom_and_velocity():
    rng = np.random.default_rng(0)
    cur = _stats(rng.normal(0, 4, 5000), rng.uniform(0, 20, 5000))       # |.|p95 ≈ 7.8
    vendor = _stats(rng.normal(0, 12, 5000), rng.uniform(0, 30, 5000))   # |.|p95 ≈ 23.5
    p = propose_multipliers(_prof(), "default", {"a": vendor}, {"a": cur})["a"]
    assert abs(p.raw_mult - vendor.abs_p95 / cur.abs_p95) < 1e-9 and p.mult == p.raw_mult and p.cap_reason == "fit"
    # max_mult cap
    p = propose_multipliers(_prof(), "default", {"a": vendor}, {"a": cur}, max_mult=2.0)["a"]
    assert p.mult == 2.0 and "max_mult" in p.cap_reason
    # headroom: joint only reaches ±15 → scaled p99 must stay inside 0.9 × 15
    p = propose_multipliers(_prof(lo=-15, hi=15), "default", {"a": vendor}, {"a": cur})["a"]
    assert p.mult < p.raw_mult and "headroom" in p.cap_reason
    assert p.mult * np.percentile(cur.values, 99) <= 0.9 * 15 + 1e-9
    # velocity cap: pipeline velocity already at the vendor's → multiplier ≤ vel_cap
    p = propose_multipliers(_prof(), "default", {"a": vendor}, {"a": cur}, pipeline={"a": vendor}, vel_cap=1.25)["a"]
    assert p.mult == pytest.approx(1.25) and "velocity" in p.cap_reason
    # skip
    assert propose_multipliers(_prof(), "default", {"a": vendor}, {"a": cur}, skip=("a",)) == {}


def test_gaze_compensation_cancels_the_linearised_elevation():
    spec = {"schema": "animacy.robot.v1", "name": "t", "display_name": "T", "description": {"urdf": "x.urdf"},
            "joints": [{"name": "b", "min": -90, "max": 90, "rest": 0, "max_speed": 100},
                       {"name": "e", "min": -90, "max": 90, "rest": 0, "max_speed": 100},
                       {"name": "w", "min": -90, "max": 90, "rest": 0, "max_speed": 100}],
            "retarget": {"default": {"b": {"mix": [{"from": "torso_lean_fwd", "gain": 1.0}, {"from": "head_x", "gain": 0.1}]},
                                     "e": {"mix": [{"from": "torso_lean_fwd", "gain": 0.4}, {"from": "mouth_open", "gain": 6.0}]},
                                     "w": {"from": "head_pitch", "gain": -1.0}}}}
    jac = {"b": (-1.0, 0.0), "e": (1.0, 0.0), "w": (-1.0, 0.0)}
    comp = gaze_compensation(Profile(**spec), "default", jac, "w", ("b", "e"))
    assert comp["torso_lean_fwd"] == pytest.approx(-1.0 + 0.4)
    assert comp["head_x"] == pytest.approx(-0.1)
    assert comp["mouth_open"] == pytest.approx(6.0)
    # cancellation: −Δb + Δe − Δw = 0 for any channel value
    for ch, g_b, g_e in [("torso_lean_fwd", 1.0, 0.4), ("head_x", 0.1, 0.0), ("mouth_open", 0.0, 6.0)]:
        assert abs(-g_b + g_e - comp[ch]) < 1e-12


TEXT = """---
schema: animacy.robot.v1
retarget:
  # a comment that must survive
  default:
    base_yaw:
      mix:
        - { from: head_yaw,  gain: -0.45 }
        - { from: torso_yaw, gain: -0.6 }   # keep me
      deadband: 0.3
    wrist_pitch:
      mix:
        - { from: head_pitch, gain: -0.9 }
        - { from: torso_lean_fwd }
      min: -85
    head_z:     { from: head_z, gain: 0.25, smooth_hz: 4 }
  puppet:
    base_yaw:    { from: shoulder_yaw, gain: -1.0, smooth_hz: 6 }
    wrist_pitch: { from: wrist_pitch, gain: -1.0 }
---
prose with gain: -0.45 in it
"""


def test_rewrite_gains_preserves_layout_and_only_touches_the_mode():
    out = rewrite_gains(TEXT, "default", {"base_yaw": {"head_yaw": -1.335}, "wrist_pitch": {"torso_lean_fwd": -0.373}, "head_z": {"head_z": 0.5}}, "2026-08-26")
    assert "# a comment that must survive" in out
    assert "- { from: head_yaw,  gain: -1.335 }  # fitted by scripts/retarget_fit.py 2026-08-26" in out
    assert "- { from: torso_yaw, gain: -0.6 }   # keep me" in out                  # untouched, no stamp
    assert "- { from: torso_lean_fwd, gain: -0.373 }  # fitted by" in out         # gain inserted where there was none
    assert "head_z:     { from: head_z, gain: 0.5, smooth_hz: 4 }  # fitted by" in out
    assert "base_yaw:    { from: shoulder_yaw, gain: -1.0, smooth_hz: 6 }" in out  # puppet untouched
    assert "wrist_pitch: { from: wrist_pitch, gain: -1.0 }" in out
    assert "prose with gain: -0.45 in it" in out
    # re-stamping replaces the old stamp instead of appending a second one
    out2 = rewrite_gains(out, "default", {"base_yaw": {"head_yaw": -2.0}}, "2027-01-01")
    line = [l for l in out2.split("\n") if "from: head_yaw" in l and "default" not in l][0]
    assert line.count("fitted by") == 1 and "2027-01-01" in line and "gain: -2.0" in line
    with pytest.raises(KeyError):
        rewrite_gains(TEXT, "default", {"base_yaw": {"nope": 1.0}}, "x")
