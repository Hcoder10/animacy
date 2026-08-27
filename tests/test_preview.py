"""`animacy preview` + the shared mesh decimator, on every robot that ships a URDF."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

yourdfpy = pytest.importorskip("yourdfpy")
pytest.importorskip("matplotlib")

from animacy import preview  # noqa: E402
from animacy.profile import load_profile, robots_root  # noqa: E402
from animacy.urdf_tools import joints_from_channels, load_urdf, pick_gaze_axis, pick_tip_link, probe, set_joints, tip_state  # noqa: E402

ROBOTS = [d for d in sorted(os.listdir(robots_root()))
          if not d.startswith("_") and os.path.exists(os.path.join(robots_root(), d, "ROBOT.md"))
          and os.path.exists(load_profile(os.path.join(robots_root(), d)).urdf_path())]


@pytest.mark.parametrize("name", ROBOTS)
def test_preview_writes_pngs_and_probe(name, tmp_path):
    out = str(tmp_path / name)
    res = preview.run(os.path.join(robots_root(), name), out=out, poses=["rest", "look_left", "look_up"], quiet=True)
    for pose in ("rest", "look_left", "look_up"):
        p = os.path.join(out, f"{pose}.png")
        assert os.path.exists(p) and os.path.getsize(p) > 10_000, pose
    prof = load_profile(os.path.join(robots_root(), name))
    assert [r["joint"] for r in res["probe"]] == prof.joint_names
    txt = open(os.path.join(out, "probe.txt"), encoding="utf-8").read()
    for j in prof.joint_names:
        assert j in txt
    js = json.load(open(os.path.join(out, "probe.json"), encoding="utf-8"))
    assert js["tip"] == res["tip"] and len(js["rows"]) == len(prof.joints)
    for r in res["probe"]:
        assert all(np.isfinite([r["dx_mm"], r["dy_mm"], r["dz_mm"], r["dyaw_deg"], r["dpitch_deg"]]))
        assert r["reads_as"]


@pytest.mark.parametrize("name", ROBOTS)
def test_calibration_poses_point_the_right_way(name):
    """The cross-robot sign contract: through each ROBOT.md, 'look left' turns the
    face left and 'look up' tips it up (canonical +yaw = left, +pitch = up)."""
    prof = load_profile(os.path.join(robots_root(), name))
    u = load_urdf(prof, load_meshes=False)
    tip = pick_tip_link(prof, u)
    gaze, _ = pick_gaze_axis(prof, u, tip)

    def yaw_pitch(channels):
        j, _ = joints_from_channels(prof, "default", channels)
        set_joints(u, prof, j)
        _, g = tip_state(u, tip, gaze)
        return np.degrees(np.arctan2(g[1], g[0])), np.degrees(np.arcsin(np.clip(g[2], -1, 1)))

    y0, p0 = yaw_pitch({})
    y1, _ = yaw_pitch({"head_yaw": 40.0})
    _, p2 = yaw_pitch({"head_pitch": 25.0})
    assert (y1 - y0 + 180) % 360 - 180 > 3.0, f"{name}: look_left did not turn the face left (dyaw {y1 - y0:.1f})"
    assert p2 - p0 > 3.0, f"{name}: look_up did not tip the face up (dpitch {p2 - p0:.1f})"


def test_probe_reads_reachy_signs():
    prof = load_profile(os.path.join(robots_root(), "reachy_mini"))
    u = load_urdf(prof, load_meshes=False)
    tip = pick_tip_link(prof, u)
    assert tip == "head"
    gaze, label = pick_gaze_axis(prof, u, tip)
    assert label == "x"
    rows = {r["joint"]: r for r in probe(prof, u, tip, gaze)}
    assert rows["head_yaw"]["dyaw_deg"] > 9 and "LEFT" in rows["head_yaw"]["reads_as"]
    assert rows["head_pitch"]["dpitch_deg"] > 9 and "UP" in rows["head_pitch"]["reads_as"]
    assert rows["head_z"]["dz_mm"] == pytest.approx(10.0, abs=1e-3)
    assert "unaffected" in rows["antenna_left"]["reads_as"]  # antennas hang off the head


def test_preview_clip_keyframes(tmp_path):
    clip = os.path.join(robots_root(), "reachy_mini", "clips", "native", "amazed1.json")
    if not os.path.exists(clip):
        pytest.skip("no native reachy clip")
    res = preview.run(os.path.join(robots_root(), "reachy_mini"), out=str(tmp_path), clip=clip, frames="2", quiet=True)
    assert len(res["pngs"]) == 2 and all(os.path.getsize(p) > 10_000 for p in res["pngs"])


def test_unknown_pose_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        preview.run(os.path.join(robots_root(), ROBOTS[0]), out=str(tmp_path), poses=["nope"], quiet=True)


def test_decimate_meshes_hits_budget_and_scales(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("fast_simplification")
    src = tmp_path / "in"
    src.mkdir()
    for i in range(3):
        trimesh.creation.icosphere(subdivisions=4, radius=100.0).export(str(src / f"ball{i}.stl"))  # 5120 faces, "mm"
    out = tmp_path / "out"
    script = os.path.join(ROOT, "scripts", "decimate_meshes.py")
    r = subprocess.run([sys.executable, script, str(src), str(out), "--budget-mb", "0.2", "--scale", "0.001"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    files = sorted(os.listdir(out))
    assert files == ["ball0.stl", "ball1.stl", "ball2.stl"]
    total = sum(os.path.getsize(out / f) for f in files)
    assert total <= 0.2 * 1024 * 1024
    m = trimesh.load(str(out / "ball0.stl"), force="mesh")
    assert len(m.faces) < 5120 and abs(m.bounds[1][0] - 0.1) < 0.01  # decimated, and mm -> m
