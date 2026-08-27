"""Structural and kinematic checks for robots/lamp/urdf/lamp.urdf.

Run:  <venv>/python -m pytest tests/test_lamp_urdf.py -q
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from animacy.export import read_autonomous_os_csv  # noqa: E402
from animacy.profile import load_profile  # noqa: E402
from animacy.retarget import to_urdf_values  # noqa: E402

yourdfpy = pytest.importorskip("yourdfpy")

ROBOT_DIR = os.path.join(ROOT, "robots", "lamp")
URDF = os.path.join(ROBOT_DIR, "urdf", "lamp.urdf")
JOINTS = ["base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch"]


@pytest.fixture(scope="module")
def urdf():
    return yourdfpy.URDF.load(URDF, load_meshes=False, build_scene_graph=True)


@pytest.fixture(scope="module")
def profile():
    return load_profile(ROBOT_DIR)


def _set_deg(u, q):
    u.update_cfg({k: math.radians(v) for k, v in q.items()})


def test_joints_exist_revolute_with_limits(urdf):
    names = [j.name for j in urdf.actuated_joints]
    assert sorted(names) == sorted(JOINTS), names
    # chain order: the pitch axle is at the neck base and the roll servo in the head (see urdf/README.md)
    assert names == ["base_yaw", "base_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll"], names
    for j in urdf.actuated_joints:
        assert j.type == "revolute", j.name
        assert j.limit is not None
        assert j.limit.lower == pytest.approx(-math.pi / 2, abs=1e-6), j.name
        assert j.limit.upper == pytest.approx(math.pi / 2, abs=1e-6), j.name
        assert np.linalg.norm(j.axis) == pytest.approx(1.0, abs=1e-6), j.name


def test_root_link_is_base(urdf):
    assert urdf.base_link == "base"


def test_meshes_resolve_and_are_small(urdf):
    base_dir = os.path.dirname(URDF)
    total = 0
    for link in urdf.link_map.values():
        assert link.visuals, link.name
        fn = link.visuals[0].geometry.mesh.filename
        assert fn.startswith("../meshes/"), fn  # relative, browser-loader friendly
        path = os.path.normpath(os.path.join(base_dir, fn))
        assert os.path.exists(path), path
        total += os.path.getsize(path)
    assert total < 6 * 1024 * 1024, total


def test_rest_pose_head_position(urdf, profile):
    """At the ROBOT.md rest pose the head sits 0.25-0.48 m up and roughly forward of the base."""
    rest = {j.name: j.rest for j in profile.joints}
    _set_deg(urdf, rest)
    head = urdf.get_transform("head", "base")[:3, 3]
    assert 0.25 <= head[2] <= 0.48, head
    # head pivot is near the vertical centreline; the lamp head (light disc) is ~6 cm further forward
    assert head[0] > -0.05, head
    look = urdf.get_transform("head", "base")[:3, :3] @ np.array([0.70, 0.0, -0.71])
    assert look[0] > 0.5, look  # looks forward (+x), not backward


def test_vendor_home_pose_is_the_cad_pose(urdf):
    """The value every vendor clip starts from reproduces the CAD assembly pose."""
    _set_deg(urdf, {"base_yaw": 0.0, "base_pitch": 29.8, "elbow_pitch": 27.1, "wrist_roll": 8.2, "wrist_pitch": -26.3})
    head = urdf.get_transform("head", "base")[:3, 3]
    assert np.allclose(head, [0.0135, 0.0, 0.3051], atol=2e-3), head
    neck = urdf.get_transform("neck", "base")[:3, 3]
    assert np.allclose(neck, [-0.0242, 0.0, 0.2636], atol=2e-3), neck


def test_sign_conventions(urdf):
    """Directions verified against the vendor's device notes (hal/drivers/tracking/constants.py)."""
    home = {"base_yaw": 0.0, "base_pitch": 29.8, "elbow_pitch": 27.1, "wrist_roll": 8.2, "wrist_pitch": -26.3}

    def look(q):
        _set_deg(urdf, q)
        return urdf.get_transform("head", "base")[:3, :3] @ np.array([0.70, 0.0, -0.71])

    l0 = look(home)
    assert look(dict(home, base_yaw=30))[1] < l0[1] - 0.2      # +yaw -> looks to the lamp's right (-y)
    assert look(dict(home, wrist_roll=50))[1] < l0[1] - 0.2    # +roll -> looks right too
    assert look(dict(home, wrist_pitch=-60))[2] > l0[2] + 0.2  # negative wrist_pitch -> looks up
    assert look(dict(home, elbow_pitch=60))[2] > l0[2] + 0.2   # +elbow -> camera up (vendor: elbow +54.8 framed the ceiling)
    assert look(dict(home, base_pitch=60))[2] < l0[2] - 0.2    # +base_pitch -> leans forward, camera down


def test_clips_keep_head_above_desk_and_in_front(urdf, profile):
    clips_dir = os.path.join(ROBOT_DIR, "clips", "native")
    for name in ["idle", "sad", "nod", "headshake", "stretching", "curious", "sleepy", "wake_up"]:
        df = read_autonomous_os_csv(os.path.join(clips_dir, f"{name}.csv"))
        vals = to_urdf_values(df, profile)
        zs, xs = [], []
        for i in range(0, len(df), 4):
            urdf.update_cfg({k: float(v[i]) for k, v in vals.items()})
            p = urdf.get_transform("head", "base")[:3, 3]
            zs.append(p[2])
            xs.append(p[0])
        assert min(zs) > 0.12, (name, min(zs))
        assert max(xs) > -0.08, (name, max(xs))
    # semantic spot checks of the convention
    df = read_autonomous_os_csv(os.path.join(clips_dir, "stretching.csv"))
    vals = to_urdf_values(df, profile)
    heights = []
    for i in range(len(df)):
        urdf.update_cfg({k: float(v[i]) for k, v in vals.items()})
        heights.append(urdf.get_transform("head", "base")[2, 3])
    assert max(heights) > heights[0] + 0.04  # stretching rises


def test_animacy_check_passes(profile):
    assert profile.check() == []
