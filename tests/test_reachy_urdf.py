"""Reachy Mini visualization URDF: structure, neutral pose, sign conventions.

Run:  python -m pytest tests/test_reachy_urdf.py -q
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RDIR = os.path.join(ROOT, "robots", "reachy_mini")

yourdfpy = pytest.importorskip("yourdfpy")

from animacy.profile import load_profile  # noqa: E402
from animacy.retarget import to_urdf_values  # noqa: E402

EXPECTED = {
    "body_yaw": "revolute", "head_x": "prismatic", "head_y": "prismatic", "head_z": "prismatic",
    "head_roll": "revolute", "head_pitch": "revolute", "head_yaw": "revolute",
    "antenna_left": "revolute", "antenna_right": "revolute",
}


@pytest.fixture(scope="module")
def profile():
    return load_profile(RDIR)


@pytest.fixture(scope="module")
def urdf(profile):
    return yourdfpy.URDF.load(profile.urdf_path(), load_meshes=False, build_scene_graph=True)


def fk(urdf, profile, **joints):
    """animacy-unit joints -> URDF config (sign/offset/units applied) -> T_base_head."""
    row = {j.name: float(joints.get(j.name, j.rest)) for j in profile.joints}
    vals = to_urdf_values(pd.DataFrame({"t": [0.0], **{k: [v] for k, v in row.items()}}), profile)
    urdf.update_cfg({k: float(v[0]) for k, v in vals.items()})
    return urdf.get_transform("head", urdf.base_link)


def test_profile_checks_clean(profile):
    assert profile.check() == []


def test_joints_exist_with_types(urdf, profile):
    for name, jtype in EXPECTED.items():
        assert name in urdf.joint_map, name
        assert urdf.joint_map[name].type == jtype, (name, urdf.joint_map[name].type)
    assert set(profile.joint_names) == set(EXPECTED)
    for j in profile.joints:
        assert j.urdf_joint in urdf.joint_map


def test_neutral_head_above_base(urdf, profile):
    T = fk(urdf, profile)
    assert 0.10 < T[2, 3] < 0.30, T[:3, 3]
    assert abs(T[0, 3]) < 1e-6 and abs(T[1, 3]) < 1e-6
    assert np.allclose(T[:3, :3], np.eye(3), atol=1e-9)
    # SDK neutral: create_head_pose() lifted by head_z_offset = 0.177 (placo_kinematics.py)
    assert abs(T[2, 3] - 0.177) < 1e-6


def test_euler_composition_matches_sdk(urdf, profile):
    roll, pitch, yaw = 10.0, 20.0, 30.0
    T = fk(urdf, profile, head_roll=roll, head_pitch=pitch, head_yaw=yaw)
    s = {j.name: j.urdf_sign for j in profile.joints}
    # SDK: create_head_pose -> Rotation.from_euler("xyz", [r, p, y]) = Rz(y) Ry(p) Rx(r), in SDK signs
    expect = R.from_euler("xyz", [roll * s["head_roll"], pitch * s["head_pitch"], yaw * s["head_yaw"]], degrees=True).as_matrix()
    assert np.abs(T[:3, :3] - expect).max() < 1e-6
    assert np.allclose(T[:3, 3], [0, 0, 0.177], atol=1e-9)  # pure rotation about the neutral origin


def test_animacy_signs(urdf, profile):
    fwd = lambda T: T[:3, 0]  # noqa: E731  head +x = gaze direction
    up = fk(urdf, profile, head_pitch=25)
    assert fwd(up)[2] > 0.3, "+head_pitch must look UP"
    left = fk(urdf, profile, head_yaw=40)
    assert fwd(left)[1] > 0.5, "+head_yaw must turn toward the robot's LEFT (+y)"
    roll = fk(urdf, profile, head_roll=20)
    assert roll[2, 1] > 0.2, "+head_roll: right ear drops, so the head's +y (left) axis tips up"
    for name, axis in (("head_x", 0), ("head_y", 1), ("head_z", 2)):
        T = fk(urdf, profile, **{name: 10.0})
        d = T[:3, 3] - np.array([0, 0, 0.177])
        assert abs(d[axis] - 0.010) < 1e-9 and np.abs(np.delete(d, axis)).max() < 1e-9, (name, d)


def test_head_pose_is_base_relative(urdf, profile):
    """SDK head poses are expressed in the base frame: body_yaw must not move the head."""
    T0 = fk(urdf, profile)
    B0 = urdf.get_transform("body", urdf.base_link)
    T1 = fk(urdf, profile, body_yaw=90)
    B1 = urdf.get_transform("body", urdf.base_link)
    assert np.allclose(T0, T1)
    # ... while the body itself turned +90 deg about z (= toward the robot's left)
    rv = R.from_matrix(B1[:3, :3] @ B0[:3, :3].T).as_rotvec()
    assert np.allclose(rv, [0, 0, np.pi / 2], atol=1e-9), rv


def test_antennas_positive_is_outward(urdf, profile):
    for name, link, sign_y in (("antenna_left", "antenna_left_link", +1), ("antenna_right", "antenna_right_link", -1)):
        fk(urdf, profile)
        J0 = urdf.get_transform(link, urdf.base_link)
        p_local = np.linalg.inv(J0) @ np.append(J0[:3, 3] + [0, 0, 0.05], 1.0)  # a point 5 cm above the hinge
        fk(urdf, profile, **{name: 90.0})
        p = (urdf.get_transform(link, urdf.base_link) @ p_local)[:3]
        dy = p[1] - J0[1, 3]
        assert sign_y * dy > 0.03, f"+{name} must swing the antenna outward (y sign {sign_y}), got dy={dy:.4f}"
        assert p[2] < J0[2, 3] + 0.03, f"+{name}=90 should bring the tip near horizontal"


def test_meshes_exist_and_budget(profile):
    text = open(profile.urdf_path(), encoding="utf-8").read()
    import re

    files = set(re.findall(r'filename="([^"]+)"', text))
    assert files, "no meshes referenced"
    base = os.path.dirname(profile.urdf_path())
    for f in files:
        assert f.startswith("../meshes/"), f
        assert os.path.exists(os.path.join(base, f)), f
    total = sum(os.path.getsize(os.path.join(RDIR, "meshes", f)) for f in os.listdir(os.path.join(RDIR, "meshes")))
    assert total < 8 * 1024 * 1024
    assert os.path.exists(os.path.join(RDIR, "meshes", "ATTRIBUTION.md"))


def test_native_clips(profile):
    d = os.path.join(RDIR, profile.native_clips.dir)
    idx = json.load(open(os.path.join(d, "index.json"), encoding="utf-8"))
    names = [c["name"] for c in idx["clips"]]
    for must in ("amazed1", "attentive1", "boredom1"):
        assert must in names
    total = 0
    for c in idx["clips"]:
        p = os.path.join(d, c["name"] + ".json")
        total += os.path.getsize(p)
        clip = json.load(open(p, encoding="utf-8"))
        assert clip["joints"] == profile.joint_names
        assert clip["rate_hz"] == 30
        n = len(clip["t"])
        for j in profile.joint_names:
            assert len(clip["data"][j]) == n
            assert min(clip["data"][j]) >= profile.joint(j).min - 1e-6
            assert max(clip["data"][j]) <= profile.joint(j).max + 1e-6
    assert total < 2 * 1024 * 1024
    # amazed1: head goes back/up and the antennas open (library semantics), in animacy signs
    am = json.load(open(os.path.join(d, "amazed1.json"), encoding="utf-8"))["data"]
    assert max(am["antenna_right"]) > 30 or max(am["antenna_left"]) > 30
