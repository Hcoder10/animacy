"""The browser retargeter must produce the same numbers as the Python one.

`web/js/retarget.js` is a port of `animacy/retarget.py:LiveRetargeter.step` and
`to_urdf_values`. This test feeds both the same profile (the JSON that
`animacy profile export` writes) and the same random channel stream, and diffs
the joint tables. Needs `node` on PATH (skipped otherwise).
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess

import numpy as np
import pytest

from animacy.profile import find_robot
from animacy.retarget import LiveRetargeter, to_urdf_values
from animacy.schema import MAPPABLE

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HARNESS = os.path.join(ROOT, "web", "dev", "retarget_parity.mjs")
NODE = shutil.which("node")

import glob

# every profile the viewer can load (web/robots/*.json), not just the headline pair
ROBOTS = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(ROOT, "web", "robots", "*.json")))
MODES = ["default", "puppet"]


def _frames(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        f = {}
        for c in MAPPABLE:
            # big, jittery, occasionally missing/NaN — exercise deadband, clamps and speed clipping
            r = rng.random()
            if r < 0.05:
                continue
            if r < 0.08:
                f[c] = float("nan")
                continue
            amp = 1.0 if c in ("brow_l", "brow_r", "brow_furrow", "eye_open_l", "eye_open_r", "mouth_open", "smile", "hand_open") else 120.0
            f[c] = amp * (0.6 * math.sin(0.3 * i + hash(c) % 7) + 0.4 * (rng.random() * 2 - 1))
        out.append(f)
    return out


def _nan_to_null(frames):
    return [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in f.items()} for f in frames]


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
@pytest.mark.parametrize("robot", ROBOTS)
@pytest.mark.parametrize("mode", MODES)
def test_js_matches_python(robot: str, mode: str):
    prof = find_robot(robot)
    if mode not in prof.retarget:
        pytest.skip(f"{robot} has no '{mode}' mode")
    profile_json = prof.to_web_json()
    dt = 1 / 30
    frames = _frames(240, seed=hash((robot, mode)) & 0xFFFF)

    job = {"profile": profile_json, "mode": mode, "dt": dt, "frames": _nan_to_null(frames)}
    res = subprocess.run([NODE, HARNESS], input=json.dumps(job), capture_output=True, text=True, cwd=ROOT, timeout=60)
    assert res.returncode == 0, res.stderr
    js = json.loads(res.stdout)

    rt = LiveRetargeter(prof, mode=mode)
    py_joints = [rt.step(f, dt) for f in frames]
    assert len(js["joints"]) == len(py_joints)
    worst = 0.0
    for i, (a, b) in enumerate(zip(js["joints"], py_joints)):
        for j in prof.joint_names:
            d = abs(a[j] - b[j])
            worst = max(worst, d)
            assert d < 1e-6, f"{robot}/{mode} frame {i} joint {j}: js={a[j]} py={b[j]}"
    # urdf conversion: run Python's on the JS joint table and compare with JS's
    import pandas as pd

    table = pd.DataFrame(js["joints"])
    py_urdf = to_urdf_values(table, prof)
    for i, row in enumerate(js["urdf"]):
        for j in prof.joints:
            assert abs(row[j.urdf_joint] - py_urdf[j.urdf_joint][i]) < 1e-9
    assert worst < 1e-6


V11_SPEC = {
    # a synthetic profile exercising every v1.1 key (docs/RETARGET.md): soft limit,
    # idle sway (explicit and default `still`), spring tracker (under- and critically damped)
    "schema": "animacy.robot.v1", "name": "v11", "display_name": "v1.1 test body", "description": {"urdf": "x.urdf"},
    "joints": [
        {"name": "a", "min": -60, "max": 60, "rest": 5, "max_speed": 300},
        {"name": "b", "min": -90, "max": 90, "rest": -10, "max_speed": 120},
        {"name": "c", "unit": "mm", "min": -30, "max": 30, "rest": 0, "max_speed": 200},
        {"name": "d", "min": -45, "max": 45, "rest": 0, "max_speed": 500},
        {"name": "e", "min": -45, "max": 45, "rest": 0, "max_speed": 60},
    ],
    "retarget": {"default": {
        "a": {"from": "head_yaw", "gain": 1.0, "soft_limit": 0.2, "spring": {"hz": 2.0, "zeta": 0.6}},
        "b": {"mix": [{"from": "head_pitch", "gain": 0.8}, {"from": "brow_l", "gain": 5}], "deadband": 0.5, "idle": {"amp": 3.0, "hz": 0.4}, "smooth_hz": 4},
        "c": {"from": "head_x", "gain": 0.3, "idle": {"amp": 2.0, "hz": 0.25, "still": 40}, "spring": {"hz": 1.5, "zeta": 1.0}, "soft_limit": 0.1},
        "d": {"from": "head_roll", "gain": 1.0, "min": -20, "max": 20, "soft_limit": 0.3},
        # over-damped spring on a slow joint: the rate limit engages, so the carried velocity is re-derived
        "e": {"from": "head_yaw", "gain": 0.8, "spring": {"hz": 3.0, "zeta": 1.5}, "settle": {"seconds": 0.6, "quiet": 0.4, "still": 12}},
    }},
}


def _v11_profile():
    from animacy.profile import Profile

    return Profile(**V11_SPEC)


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
@pytest.mark.parametrize("kind", ["jittery", "still-then-move"])
def test_js_matches_python_v11_features(kind: str):
    """soft limit + idle sway + spring tracker: same numbers to 1e-6 (idle only
    switches on while the target is still, so the second stream holds for 3 s)."""
    prof = _v11_profile()
    dt = 1 / 30
    if kind == "jittery":
        frames = _frames(240, seed=11)
    else:
        frames = [{"head_yaw": 3.0, "head_pitch": -2.0, "head_x": 5.0, "head_roll": 1.0, "brow_l": 0.1}] * 90
        frames += _frames(90, seed=12)
    job = {"profile": prof.to_web_json(), "mode": "default", "dt": dt, "frames": _nan_to_null(frames)}
    res = subprocess.run([NODE, HARNESS], input=json.dumps(job), capture_output=True, text=True, cwd=ROOT, timeout=60)
    assert res.returncode == 0, res.stderr
    js = json.loads(res.stdout)["joints"]
    rt = LiveRetargeter(prof, mode="default")
    py = [rt.step(f, dt) for f in frames]
    worst = {j: 0.0 for j in prof.joint_names}
    for i, (a, b) in enumerate(zip(js, py)):
        for j in prof.joint_names:
            d = abs(a[j] - b[j])
            worst[j] = max(worst[j], d)
            assert d < 1e-6, f"v1.1/{kind} frame {i} joint {j}: js={a[j]} py={b[j]}"
    # sanity: the idle joints actually swayed (else the test proves nothing about idle)
    if kind == "still-then-move":
        b_vals = [p["b"] for p in py[30:90]]
        assert max(b_vals) - min(b_vals) > 0.5, "idle sway did not engage on joint b while still"


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_exported_profile_json_is_what_the_page_loads():
    """web/robots/<name>.json must be a current export of robots/<name>/ROBOT.md."""
    for robot in ROBOTS:
        path = os.path.join(ROOT, "web", "robots", f"{robot}.json")
        assert os.path.exists(path), f"missing {path}: run animacy profile export robots/{robot} -o {path}"
        on_disk = json.load(open(path, encoding="utf-8"))
        fresh = json.loads(json.dumps(find_robot(robot).to_web_json()))
        assert on_disk == fresh, f"{path} is stale — re-run: animacy profile export robots/{robot} -o web/robots/{robot}.json"


def test_manifest_matches_disk():
    """web/manifest.json must reflect the URDFs and clips that exist."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_manifest", os.path.join(ROOT, "web", "dev", "build_manifest.py"))
    bm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bm)
    fresh = bm.build()
    on_disk = json.load(open(os.path.join(ROOT, "web", "manifest.json"), encoding="utf-8"))
    for k in ("robots", "native", "clips", "models", "bundle"):
        assert on_disk.get(k) == fresh[k], f"web/manifest.json '{k}' is stale — run python web/dev/build_manifest.py"
    for name, r in fresh["robots"].items():
        assert r["exists"], f"{name}: {r['urdf']} missing (viewer would fall back to the dev stand-in)"
