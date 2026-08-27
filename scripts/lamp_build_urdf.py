"""Generate robots/lamp/urdf/lamp.urdf for the Autonomous Lamp.

Geometry (joint pivots, link meshes) comes from the vendor's ``lamp.glb``
(see ``lamp_extract_meshes.py``). The CAD assembly pose ("bind pose") is the
pose the meshes are exported in; ``BIND_Q`` says which *vendor joint values*
(degrees, the numbers in ``clips/native/*.csv``) that pose corresponds to, and
``AXIS`` gives the URDF joint axis in the bind frame, signed so that a positive
vendor value rotates the way the vendor's clips imply. See
``robots/lamp/urdf/README.md`` for how each of these was derived.

Every link frame = origin at that link's joint pivot, axes = world axes at the
bind pose. So each mesh sits at identity in its link and each joint origin is
``translation(parent pivot -> child pivot) * Rot(axis, -bind_q)`` so that
feeding the vendor's value (deg -> rad, sign +1, offset 0) reproduces the CAD
pose at ``BIND_Q`` and the vendor's zero at 0.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "robots", "lamp", "urdf", "lamp.urdf")

# Joint pivots in URDF world frame at the bind pose (metres), from the GLB armature.
PIV = {
    "base_yaw":    np.array([0.0,     0.0, 0.0495]),
    "base_pitch":  np.array([0.0,     0.0, 0.0966]),
    "elbow_pitch": np.array([-0.0948, 0.0, 0.1952]),
    "wrist_pitch": np.array([-0.0242, 0.0, 0.2636]),
    "wrist_roll":  np.array([0.0135,  0.0, 0.3051]),
}
_neck = PIV["wrist_roll"] - PIV["wrist_pitch"]
NECK_AXIS = _neck / np.linalg.norm(_neck)          # (0.672, 0, 0.740): up-forward along the neck

# Vendor joint values (deg) at the CAD/bind pose. The CAD pose is the vendor's
# home pose: the value every clip starts from / returns to (base_yaw is the
# clips' -2.4 median, treated as 0 because the CAD arm plane is exactly x-z).
BIND_Q = {"base_yaw": 0.0, "base_pitch": 29.8, "elbow_pitch": 27.1, "wrist_pitch": -26.3, "wrist_roll": 8.2}

# URDF axis (bind frame) per joint; sign = the direction a POSITIVE vendor value turns.
AXIS = {
    "base_yaw":    np.array([0.0, 0.0, -1.0]),   # + = turn to the lamp's right (clockwise from above)
    "base_pitch":  np.array([0.0, 1.0, 0.0]),    # + = lower arm leans forward
    "elbow_pitch": np.array([0.0, -1.0, 0.0]),   # + = fold closes (opposite sense to base_pitch, vendor ELBOW_PITCH_SIGN=-1)
    "wrist_pitch": np.array([0.0, 1.0, 0.0]),    # + = head tips down (negative = look up)
    "wrist_roll":  -NECK_AXIS,                   # + = head pans to the lamp's right (same sense as base_yaw)
}

CHAIN = [  # joint, parent link, child link, mesh
    ("base_yaw", "base", "swivel", "swivel"),
    ("base_pitch", "swivel", "lower_arm", "lower_arm"),
    ("elbow_pitch", "lower_arm", "upper_arm", "upper_arm"),
    ("wrist_pitch", "upper_arm", "neck", "neck"),
    ("wrist_roll", "neck", "head", "head"),
]
MASS = {"base": 0.9, "swivel": 0.12, "lower_arm": 0.10, "upper_arm": 0.14, "neck": 0.05, "head": 0.18}
COLOR = {"base": "0.93 0.90 0.84 1", "swivel": "0.93 0.90 0.84 1", "lower_arm": "0.93 0.90 0.84 1",
         "upper_arm": "0.93 0.90 0.84 1", "neck": "0.35 0.35 0.35 1", "head": "0.93 0.90 0.84 1"}
LIMIT = math.pi / 2


def rot_axis(a, th):
    a = np.asarray(a, float); a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def rpy_from_R(R):
    """URDF rpy (extrinsic x-y-z, i.e. R = Rz(y) Ry(p) Rx(r))."""
    sy = -R[2, 0]
    p = math.asin(max(-1.0, min(1.0, sy)))
    if abs(math.cos(p)) > 1e-9:
        r = math.atan2(R[2, 1], R[2, 2]); y = math.atan2(R[1, 0], R[0, 0])
    else:
        r = math.atan2(-R[1, 2], R[1, 1]); y = 0.0
    return r, p, y


def fmt(v):
    return " ".join(f"{x:.6g}" if abs(x) > 1e-9 else "0" for x in v)


def link_xml(name, mesh):
    m = MASS[name]
    i = 1e-4 * m
    return f"""  <link name="{name}">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="../meshes/{mesh}.stl" scale="1 1 1"/></geometry>
      <material name="{name}_mat"><color rgba="{COLOR[name]}"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="../meshes/{mesh}.stl" scale="1 1 1"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{m}"/>
      <inertia ixx="{i:.3g}" ixy="0" ixz="0" iyy="{i:.3g}" iyz="0" izz="{i:.3g}"/>
    </inertial>
  </link>
"""


def joint_xml(joint, parent, child):
    p_parent = PIV[CHAIN[[c[0] for c in CHAIN].index(joint) - 1][0]] if joint != "base_yaw" else np.zeros(3)
    xyz = PIV[joint] - p_parent
    a = AXIS[joint]
    R = rot_axis(a, -math.radians(BIND_Q[joint]))
    r, p, y = rpy_from_R(R)
    return f"""  <joint name="{joint}" type="revolute">
    <origin xyz="{fmt(xyz)}" rpy="{fmt((r, p, y))}"/>
    <parent link="{parent}"/>
    <child link="{child}"/>
    <axis xyz="{fmt(a)}"/>
    <limit lower="{-LIMIT:.6f}" upper="{LIMIT:.6f}" effort="3.0" velocity="4.36"/>
  </joint>
"""


def build() -> str:
    out = ['<?xml version="1.0"?>',
           '<!-- Autonomous Lamp (Autonomous OS, LeLamp-derived 5x STS3215 desk lamp). -->',
           '<!-- Generated by scripts/lamp_build_urdf.py from the vendor CAD (lamp.glb); see urdf/README.md. -->',
           '<!-- Joint values are the vendor\'s own (hal ServoMoveRequest / recordings CSV), degrees -> radians, no sign or offset. -->',
           '<robot name="autonomous_lamp">']
    out.append(link_xml("base", "base"))
    for joint, parent, child, mesh in CHAIN:
        out.append(joint_xml(joint, parent, child))
        out.append(link_xml(child, mesh))
    out.append("</robot>\n")
    return "\n".join(out)


def main() -> int:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(build())
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
