"""v1.1 retarget features (docs/RETARGET.md): soft limit, spring tracker, idle
sway, gaze compensation, antenna symmetry — and the doc's numeric example."""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from animacy.profile import Profile, load_profile, robots_root
from animacy.retarget import (
    IDLE_RELEASE_S, LiveRetargeter, idle_phase, idle_value, raw_joint_targets, retarget_clip, soft_clip, spring_lag_s,
    spring_step,
)
from animacy.schema import HumanClip, empty_frames

DT = 1 / 30


def _spec(**mapping):
    return {"schema": "animacy.robot.v1", "name": "t", "display_name": "T", "description": {"urdf": "x.urdf"},
            "joints": [{"name": "a", "min": -60, "max": 60, "rest": 0, "max_speed": 300}],
            "retarget": {"default": {"a": mapping}}}


# ---------------------------------------------------------------- soft limit
def test_soft_clip_identity_inside_knee_monotonic_and_bounded():
    lo, hi, frac = -60.0, 60.0, 0.2
    x = np.linspace(-200, 200, 4001)
    y = soft_clip(x, lo, hi, frac)
    inside = np.abs(x) <= 36
    assert np.allclose(y[inside], x[inside])
    assert np.all(np.diff(y) > 0)
    assert y.max() < hi and y.min() > lo
    # C1 at the knee: slope just outside ≈ 1
    assert abs((soft_clip(36.1, lo, hi, frac) - soft_clip(35.9, lo, hi, frac)) / 0.2 - 1.0) < 1e-3
    assert soft_clip(50.0, lo, hi, None) == 50.0 and soft_clip(50.0, lo, hi, 0.0) == 50.0


# ---------------------------------------------------------------- spring
def _step_response(hz, zeta, n=300, target=10.0):
    y, v, ys = 0.0, 0.0, []
    for _ in range(n):
        y, v = spring_step(y, v, target, DT, hz, zeta)
        ys.append(y)
    return np.array(ys)


def test_spring_overshoot_matches_damping_ratio():
    crit = _step_response(3.0, 1.0)
    assert crit.max() <= 10.0 + 1e-6 and abs(crit[-1] - 10.0) < 1e-3
    under = _step_response(3.0, 0.45)
    expected = math.exp(-math.pi * 0.45 / math.sqrt(1 - 0.45 ** 2))  # 20.5 %
    assert abs((under.max() - 10.0) / 10.0 - expected) < 0.03
    assert abs(under[-1] - 10.0) < 1e-2


def test_spring_is_exact_in_every_damping_regime():
    """Coefficients equal the matrix exponential of the state matrix (ZOH), so
    the tracker is stable for any hz/dt — including profile.check()'s rate/4."""
    expm = pytest.importorskip("scipy.linalg").expm
    from animacy.retarget import spring_coefficients

    for hz in (0.5, 3.0, 7.5, 12.0):
        for zeta in (0.2, 0.7, 1.0, 1.3, 2.0):
            w = 2 * math.pi * hz
            M = expm(np.array([[0.0, 1.0], [-w * w, -2 * zeta * w]]) * DT)
            pp, pv, vp, vv = spring_coefficients(hz, zeta, DT)
            assert np.abs(M - np.array([[pp, pv], [vp, vv]])).max() < 1e-9, (hz, zeta)
    y = _step_response(7.5, 0.3, n=600)
    assert np.isfinite(y).all() and abs(y[-1] - 10.0) < 1e-6
    assert abs((y.max() - 10.0) / 10.0 - math.exp(-math.pi * 0.3 / math.sqrt(1 - 0.09))) < 0.01


def test_live_spring_respects_rate_limit_and_limits():
    p = Profile(**_spec(**{"from": "head_yaw", "gain": 1.0, "spring": {"hz": 6.0, "zeta": 0.3}}))
    rt = LiveRetargeter(p)
    prev = 0.0
    for i in range(120):
        y = rt.step({"head_yaw": 200.0 if i < 60 else -200.0}, DT)["a"]
        assert abs(y - prev) <= 300 * DT + 1e-9
        assert -60 <= y <= 60
        assert abs(rt.vel["a"] - (y - prev) / DT) < 1e-9   # velocity re-derived after the clamps
        prev = y


def test_offline_spring_lag_is_compensated():
    """A 0.5 Hz sine through the offline spring path lands within a frame of the
    input, while the live path lags by ≈ ζ/(π·hz)."""
    hz, zeta = 3.0, 0.7
    p = Profile(**_spec(**{"from": "head_yaw", "gain": 1.0, "spring": {"hz": hz, "zeta": zeta}}))
    f = empty_frames(300)
    f["face_valid"] = 1.0
    t = f["t"].to_numpy()
    x = 20 * np.sin(2 * np.pi * 0.5 * t)
    f["head_yaw"] = x
    clip = HumanClip.from_frames(f, source="synthetic")
    off = retarget_clip(clip, p)["a"].to_numpy()
    rt = LiveRetargeter(p)
    live = np.array([rt.step({"head_yaw": float(v)}, DT)["a"] for v in x])

    def lag_frames(y):
        best, best_c = 0, -np.inf
        for d in range(0, 8):
            c = np.dot(y[60 + d:], x[60:len(x) - d]) if d else np.dot(y[60:], x[60:])
            if c > best_c:
                best, best_c = d, c
        return best

    assert lag_frames(off) <= 1
    expect = spring_lag_s(p.mapping()["a"].spring) / DT
    assert abs(lag_frames(live) - expect) <= 1.5


# ---------------------------------------------------------------- idle
def test_idle_generator_is_deterministic_and_bounded():
    vals = np.array([idle_value(i * DT, 3.0, 0.4, 1) for i in range(3000)])
    assert np.abs(vals).max() <= 3.0 + 1e-9
    assert np.abs(vals).max() > 2.0            # actually sways
    assert idle_value(1.234, 3.0, 0.4, 1) == idle_value(1.234, 3.0, 0.4, 1)
    assert len({round(idle_phase(j, k), 9) for j in range(9) for k in range(3)}) == 27


def test_idle_engages_when_still_and_fades_when_moving():
    with_idle = Profile(**_spec(**{"from": "head_yaw", "gain": 1.0, "idle": {"amp": 3.0, "hz": 0.4}, "smooth_hz": 4}))
    without = Profile(**_spec(**{"from": "head_yaw", "gain": 1.0, "smooth_hz": 4}))
    a, b = LiveRetargeter(with_idle), LiveRetargeter(without)
    still = [(a.step({"head_yaw": 5.0}, DT)["a"], b.step({"head_yaw": 5.0}, DT)["a"]) for _ in range(150)]
    diff = np.array([x - y for x, y in still[60:]])
    assert 0.5 < np.abs(diff).max() <= 3.0 + 1e-6
    # now move: gate closes within a frame (activity envelope attacks instantly)
    moving = [(a.step({"head_yaw": 5.0 + 20 * math.sin(0.5 * i)}, DT)["a"], b.step({"head_yaw": 5.0 + 20 * math.sin(0.5 * i)}, DT)["a"]) for i in range(60)]
    diff_m = np.array([x - y for x, y in moving[5:]])
    assert np.abs(diff_m).max() < 0.5
    # release: after ~3 time constants of stillness the sway is back
    for _ in range(int(3 * IDLE_RELEASE_S / DT) + 30):
        a.step({"head_yaw": 5.0}, DT)
        b.step({"head_yaw": 5.0}, DT)
    back = np.array([a.step({"head_yaw": 5.0}, DT)["a"] - b.step({"head_yaw": 5.0}, DT)["a"] for _ in range(90)])
    assert np.abs(back).max() > 0.5


def test_offline_and_live_idle_use_the_same_generator():
    p = Profile(**_spec(**{"from": "head_yaw", "gain": 1.0, "idle": {"amp": 2.0, "hz": 0.3}, "spring": {"hz": 2.0, "zeta": 1.0}}))
    f = empty_frames(240)
    f["face_valid"] = 1.0
    f["head_yaw"] = 4.0
    clip = HumanClip.from_frames(f, source="synthetic")
    off = retarget_clip(clip, p)["a"].to_numpy()
    rt = LiveRetargeter(p)
    live = np.array([rt.step({"head_yaw": 4.0}, DT)["a"] for _ in range(240)])
    # same sway once both trackers have settled on the constant part (offline starts at the target, live at rest)
    assert np.abs(off[120:] - live[120:]).max() < 0.05
    assert np.ptp(off[120:]) > 1.0


# ---------------------------------------------------------------- profile checks
def test_check_rejects_unstable_spring_and_huge_idle():
    p = Profile(**_spec(**{"from": "head_yaw", "spring": {"hz": 9.0, "zeta": 0.7}}))
    assert any("spring.hz" in e for e in p.check())
    p = Profile(**_spec(**{"from": "head_yaw", "idle": {"amp": 40.0, "hz": 0.3}}))
    assert any("idle.amp" in e for e in p.check())
    with pytest.raises(Exception):
        Profile(**_spec(**{"from": "head_yaw", "soft_limit": 0.6}))


def test_v1_profile_without_new_keys_is_unchanged():
    """No spring/idle/soft_limit → exactly the v1 one-pole + clamp arithmetic."""
    p = Profile(**_spec(**{"from": "head_yaw", "gain": 1.0, "smooth_hz": 6}))
    rt = LiveRetargeter(p)
    prev = 0.0
    for _ in range(5):
        y = rt.step({"head_yaw": 8.0}, DT)["a"]     # 8 < 300/30 per frame: the rate limit stays out of it
        alpha = 1 - math.exp(-2 * math.pi * 6 * DT)
        assert abs(y - (prev + alpha * (8.0 - prev))) < 1e-12
        prev = y


# ---------------------------------------------------------------- real profiles
ROBOTS = [d for d in sorted(os.listdir(robots_root())) if not d.startswith("_") and os.path.exists(os.path.join(robots_root(), d, "ROBOT.md"))]


@pytest.mark.parametrize("name", ROBOTS)
def test_live_path_is_legal_on_a_jittery_stream(name):
    p = load_profile(os.path.join(robots_root(), name))
    rt = LiveRetargeter(p)
    rng = np.random.default_rng(0)
    prev = {j.name: j.rest for j in p.joints}
    for i in range(300):
        ch = {c: float(rng.normal(0, 60)) for c in ("head_yaw", "head_pitch", "head_roll", "head_x", "head_y", "head_z", "torso_lean_fwd", "torso_yaw")}
        ch.update({c: float(rng.uniform(0, 1)) for c in ("brow_l", "brow_r", "brow_furrow", "mouth_open", "smile")})
        out = rt.step(ch, DT)
        for j in p.joints:
            assert j.min - 1e-9 <= out[j.name] <= j.max + 1e-9, (name, j.name)
            assert abs(out[j.name] - prev[j.name]) <= j.max_speed * DT + 1e-9, (name, j.name)
        prev = out


def test_lamp_gaze_stays_on_the_person_under_lean():
    pytest.importorskip("yourdfpy")
    from animacy.retarget_fit import gaze_error_cases, load_urdf

    p = load_profile(os.path.join(robots_root(), "lamp"))
    urdf = load_urdf(p)
    cases = [{"head_x": 100.0}, {"torso_lean_fwd": 20.0}, {"mouth_open": 1.0}, {"head_z": 50.0},
             {"head_x": 100.0, "torso_lean_fwd": 15.0, "head_z": 30.0, "mouth_open": 0.5}]
    for ch, e, _ in gaze_error_cases(p, "default", urdf, cases):
        assert abs(e) < 1.0, (ch, e)


def test_lamp_compensation_terms_are_consistent_with_the_planar_chain():
    """The `tag: gaze_comp` wrist_pitch term for every channel feeding base/elbow must be −g_base + g_elbow."""
    from animacy.retarget_fit import gaze_comp_terms

    p = load_profile(os.path.join(robots_root(), "lamp"))
    mp = p.mapping("default")
    base = {t.from_: t.gain for t in mp["base_pitch"].terms()}
    elbow = {t.from_: t.gain for t in mp["elbow_pitch"].terms()}
    comp = gaze_comp_terms(p, "default", "wrist_pitch")
    for ch in set(base) | set(elbow):
        assert ch in comp, f"wrist_pitch lacks a `tag: gaze_comp` term for {ch}"
        assert abs(comp[ch] - (-base.get(ch, 0.0) + elbow.get(ch, 0.0))) < 1e-3, ch   # gains are written to 5 significant digits


def test_settle_numeric_example_in_docs():
    """docs/RETARGET.md §settle: joint rest 0, ±60, 300/s, from head_yaw gain 1,
    settle {0.6, 0.4, still 12}; head_yaw = 20 held, then 40 at frame 32."""
    spec = _spec(**{"from": "head_yaw", "gain": 1.0, "settle": {"seconds": 0.6, "quiet": 0.4, "still": 12.0}})
    p = Profile(**spec)
    rt = LiveRetargeter(p)
    us = []
    for i in range(40):
        rt.step({"head_yaw": 20.0 if i < 31 else 40.0}, DT)
        us.append((rt.quiet["a"], rt.blend["a"]))
    q, b = us[0]
    assert q == 0.0 and b == 0.0                       # frame 1: a = 600 ≥ 12 → quiet resets
    assert abs(us[12][0] - 12 * DT) < 1e-9 and us[12][1] < 1e-12  # frame 13: quiet = 0.4 = Q → b still 0
    assert abs(us[13][1] - (13 * DT - 0.4) / 0.6) < 1e-9          # frame 14: b = 0.0556
    assert abs(us[30][1] - 1.0) < 1e-9                             # frame 31: quiet 1.0 → fully settled
    assert abs(us[31][1] - (1.0 - 4 * DT / 0.6)) < 1e-9 and us[31][0] == 0.0   # frame 32: motion → release 0.7778
    assert abs(us[34][1] - (1.0 - 16 * DT / 0.6)) < 1e-9          # frame 35: 0.1111
    assert us[35][1] == 0.0                                        # frame 36: released
    # offline path uses the same rule: a still clip settles to rest after Q + S
    f = empty_frames(150)
    f["face_valid"] = 1.0
    f["head_yaw"] = 20.0
    f["speaking"] = 0.0
    off = retarget_clip(HumanClip.from_frames(f, source="synthetic"), p)["a"].to_numpy()
    assert abs(off[-1]) < 0.05 and off[5] > 15.0
    f["speaking"] = 1.0                                             # talking: never settles
    off = retarget_clip(HumanClip.from_frames(f, source="synthetic"), p)["a"].to_numpy()
    assert off[-1] > 19.0


def test_reachy_antennas_splay_symmetrically_and_droop_on_furrow():
    p = load_profile(os.path.join(robots_root(), "reachy_mini"))

    def q(**ch):
        f = empty_frames(1)
        for k, v in ch.items():
            f[k] = v
        r = raw_joint_targets(f, p, "default").iloc[0]
        return float(r["antenna_left"]), float(r["antenna_right"])

    l, r = q(brow_l=1.0, brow_r=1.0)
    assert r > 30 and abs(l + r) < 1e-6          # symmetric outward splay (right +, left −)
    l, r = q(brow_furrow=1.0)
    assert r > 60 and abs(l + r) < 1e-6          # ears down: bigger splay, still symmetric
    l, r = q(head_roll=10.0)
    assert abs(l - r) < 1e-6 and l < 0           # common-mode counter-tilt
    l, r = q(brow_l=1.0)
    assert l < -30 and abs(r) < 1e-6             # one brow → one ear


# ---------------------------------------------------------------- docs/RETARGET.md numeric example
def test_numeric_example_in_docs():
    """The worked example in docs/RETARGET.md (joint a: rest 0, ±60, max_speed
    150; from head_yaw gain 1, soft_limit 0.2, spring 2 Hz ζ 0.6; head_yaw = 50)."""
    from animacy.retarget import spring_coefficients

    spec = _spec(**{"from": "head_yaw", "gain": 1.0, "soft_limit": 0.2, "spring": {"hz": 2.0, "zeta": 0.6}})
    spec["joints"][0]["max_speed"] = 150
    rt = LiveRetargeter(Profile(**spec))
    u = soft_clip(50.0, -60, 60, 0.2)
    assert abs(u - 48.60201) < 1e-5
    pp, pv, vp, vv = spring_coefficients(2.0, 0.6, DT)
    assert np.allclose([pp, pv, vp, vv], [0.926342, 0.025443, -4.017812, 0.542669], atol=1e-6)
    y1 = rt.step({"head_yaw": 50.0}, DT)["a"]
    assert abs(y1 - 3.57994) < 1e-5 and abs(rt.vel["a"] - 195.2738) < 1e-3   # nothing clipped: the spring's own v
    y2 = rt.step({"head_yaw": 50.0}, DT)["a"]
    # spring wants 11.864 (Δ 8.28 > 150/30 = 5): rate limit → 8.57994, v re-derived = 150
    assert abs(y2 - 8.57994) < 1e-5 and abs(rt.vel["a"] - 150.0) < 1e-9
    # idle example: joint index 1, amp 3, hz 0.4, t = 1/30
    assert np.allclose([idle_phase(1, k) for k in range(3)], [3.316668, 5.716631, 1.833409], atol=1e-6)
    assert abs(idle_value(1 / 30, 3.0, 0.4, 1) - (-0.211154)) < 1e-6
    assert abs(idle_value(1.0, 3.0, 0.4, 1) - (-0.513490)) < 1e-6
