"""SO-101 profile: ROBOT.md checks, URDF/joint contract, mapping directions in sim.

Run:  python -m pytest tests/test_so101.py -q
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RDIR = os.path.join(ROOT, "robots", "so101")

yourdfpy = pytest.importorskip("yourdfpy")

from animacy.profile import load_profile  # noqa: E402
from animacy.retarget import raw_joint_targets, to_urdf_values  # noqa: E402
from animacy.schema import CHANNELS  # noqa: E402

LEROBOT_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TIP = "gripper_frame_link"


@pytest.fixture(scope="module")
def profile():
    return load_profile(RDIR)


@pytest.fixture(scope="module")
def urdf(profile):
    return yourdfpy.URDF.load(profile.urdf_path(), load_meshes=False, build_scene_graph=True)


def tip_pose(urdf, profile, mode="default", **channels):
    row = {c: 0.0 for c in CHANNELS}
    row.update(channels)
    row["t"] = 0.0
    table = raw_joint_targets(pd.DataFrame({k: [v] for k, v in row.items()}), profile, mode)
    vals = to_urdf_values(table, profile)
    urdf.update_cfg({k: float(v[0]) for k, v in vals.items()})
    T = urdf.get_transform(TIP, urdf.base_link)
    return T[:3, 3], T[:3, 2], {j: float(table[j].iloc[0]) for j in profile.joint_names}


def test_profile_checks_clean(profile):
    assert profile.check() == []
    assert profile.joint_names == LEROBOT_JOINTS
    assert set(profile.retarget) == {"default", "puppet"}
    assert profile.export.formats == ["lerobot"]


def test_urdf_joints_and_limits(urdf, profile):
    for j in profile.joints:
        uj = urdf.joint_map[j.urdf_joint]
        assert uj.type == "revolute"
        # profile limits (deg) sit inside the URDF's (rad)
        assert np.degrees(uj.limit.lower) - 1e-6 <= j.min and j.max <= np.degrees(uj.limit.upper) + 1e-6, j.name
    text = open(profile.urdf_path(), encoding="utf-8").read()
    files = set(re.findall(r'filename="([^"]+)"', text))
    assert files and all(f.startswith("../meshes/") for f in files)
    base = os.path.dirname(profile.urdf_path())
    for f in files:
        assert os.path.exists(os.path.join(base, f)), f
    mdir = os.path.join(RDIR, "meshes")
    assert sum(os.path.getsize(os.path.join(mdir, f)) for f in os.listdir(mdir)) < 6 * 1024 * 1024
    assert os.path.exists(os.path.join(mdir, "ATTRIBUTION.md"))


def test_rest_is_attentive(urdf, profile):
    p, gaze, _ = tip_pose(urdf, profile)
    assert 0.15 < p[0] < 0.30 and abs(p[1]) < 0.01 and 0.22 < p[2] < 0.35, p
    assert gaze[0] > 0.9 and -0.1 < gaze[2] < 0.4, gaze  # level, slightly up


def test_default_mapping_directions(urdf, profile):
    p0, g0, _ = tip_pose(urdf, profile)
    p, g, j = tip_pose(urdf, profile, head_yaw=60)
    assert p[1] > 0.08 and g[1] > 0.5, "look left must swing the gripper to the robot's left (+y)"
    assert j["shoulder_pan"] < 0
    p, g, _ = tip_pose(urdf, profile, head_pitch=40)
    assert g[2] > g0[2] + 0.2, "look up must tip the gripper up"
    p, g, _ = tip_pose(urdf, profile, torso_lean_fwd=30, head_x=60)
    assert p[0] > p0[0] + 0.02, "lean in must reach forward"
    _, _, j = tip_pose(urdf, profile, mouth_open=1.0)
    assert j["gripper"] > 30
    _, _, j = tip_pose(urdf, profile, brow_l=1.0, brow_r=1.0)
    assert j["wrist_flex"] < 30 - 5


def test_puppet_mapping_is_identity_on_the_arm(urdf, profile):
    _, _, j = tip_pose(urdf, profile, mode="puppet", shoulder_yaw=20, shoulder_pitch=150, elbow_flex=70, wrist_pitch=20,
                       wrist_roll=15, hand_open=0.9)
    assert j["shoulder_pan"] == pytest.approx(-20)
    assert j["shoulder_lift"] == pytest.approx(90 - 150)
    assert j["elbow_flex"] == pytest.approx(-70)  # URDF + = folds down; human flexion = hand up
    assert j["wrist_flex"] == pytest.approx(-20)
    assert j["wrist_roll"] == pytest.approx(15)
    assert j["gripper"] == pytest.approx(90)
    # arm hanging straight down would hit the table: clamped by the mapping's max
    _, _, j = tip_pose(urdf, profile, mode="puppet", shoulder_pitch=0)
    assert j["shoulder_lift"] == pytest.approx(40)


def test_web_export_present():
    p = os.path.join(ROOT, "web", "robots", "so101.json")
    if not os.path.exists(p):
        pytest.skip("web/robots/so101.json not exported")
    d = json.load(open(p, encoding="utf-8"))
    assert d["name"] == "so101" and [j["name"] for j in d["joints"]] == LEROBOT_JOINTS
    m = os.path.join(ROOT, "web", "manifest.json")
    if os.path.exists(m):
        man = json.load(open(m, encoding="utf-8"))
        assert man["robots"]["so101"]["exists"] is True
