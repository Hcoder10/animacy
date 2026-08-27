#!/usr/bin/env python
"""Build ``robots/reachy_mini/urdf/reachy_mini.urdf`` from Pollen's vendor URDF.

Pollen's ``robot.urdf`` is the real mechanism: a 6-actuator Stewart platform
with passive spherical chains closing on the head link ``xl_330``. Browser
URDF loaders (and animacy's joint tables) need a *serial* chain whose joints
are the SDK's control variables instead. This script derives one from the
vendor file so nothing is hand-typed:

    base --body_yaw--> body  (vendor ``yaw_body``, same origin/axis/limits)
    base --head_x/head_y/head_z (prismatic, metres)--> ... at the SDK neutral
         --head_yaw(z)--> --head_pitch(y)--> --head_roll(x)--> head
    head --antenna_right / antenna_left (revolute)--> antenna links
    body --fixed--> stewart_static (motor horns + rods posed at the neutral)

Frames and numbers, with their sources:

* World = vendor root link ``body_foot_3dprint``. Placo/analytical IK in the
  ``reachy_mini`` SDK expresses the commanded head pose in this frame
  (``PlacoKinematics.head_frame`` task, ``AnalyticalKinematics.ik``) and the
  body yaw is an independent joint: the head does NOT rotate with the body.
  That is why the head chain hangs off ``base``, not ``body``.
* Neutral head pose = ``create_head_pose()`` = identity, which the SDK lifts by
  ``head_z_offset = 0.177`` m (``placo_kinematics.py``, ``kinematics_data.json``)
  before solving. So the ``head`` frame at all-zero joints is
  ``T = translate(0, 0, 0.177)`` in the world frame, x forward / y left / z up
  (camera sits at +x, right antenna at -y - checked from the vendor FK).
* Rotation order: ``create_head_pose`` uses ``Rotation.from_euler("xyz")`` =
  ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``. A serial chain about moving axes composes
  left-to-right, so the chain is yaw -> pitch -> roll. Joint values are the
  SDK's radians (ROS body frame: +pitch = nose DOWN). ``ROBOT.md`` carries the
  animacy sign (``urdf_sign: -1`` on ``head_pitch``).
* Antennas: the vendor model's antenna joint sign is the negative of the real
  robot's (``daemon/backend/mujoco/backend.py`` writes ``ctrl = -target`` and
  reads ``-qpos``). This URDF flips the axis (``0 0 -1`` in the vendor horn
  frame) so its joint values equal the SDK's real-robot radians directly.
* Static Stewart parts: motor angles from the SDK's analytical IK at the
  neutral pose (``+-35.90 deg`` alternating); each rod is placed from the
  motor-arm ball to the head's attachment point (distance must equal the
  vendor rod length, 0.085 m - checked). They do not follow the head.

Run:  python scripts/reachy_build_urdf.py  [--decimate 0.5] [--skip-meshes]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np
import yourdfpy
from scipy.spatial.transform import Rotation as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RDIR = os.path.join(ROOT, "robots", "reachy_mini")
VENDOR_URDF = os.path.join(RDIR, "vendor", "urdf", "robot_no_collision.urdf")
VENDOR_ASSETS = os.path.join(RDIR, "vendor", "urdf", "assets")
OUT_URDF = os.path.join(RDIR, "urdf", "reachy_mini.urdf")
MESH_DIR = os.path.join(RDIR, "meshes")

HEAD_Z_OFFSET = 0.177          # reachy_mini SDK: placo_kinematics.py / kinematics_data.json
# Fallback if the SDK is not importable: AnalyticalKinematics().ik(np.eye(4)) on SDK 1.9.0.
STEWART_Q_NEUTRAL_FALLBACK = [0.6265, -0.6265, 0.6265, -0.6265, 0.6265, -0.6265]
# Screws / connectors: invisible detail, ~1.3 MB of STL. Dropped to stay under the 8 MB budget.
SKIP_MESH_PREFIXES = ("phs_1_7x20_5", "bts2_m2_6x8", "b3b_eh")
MESH_BUDGET_BYTES = 8 * 1024 * 1024

HEAD_LIMITS = {  # generous viewer limits (rad / m); ROBOT.md's limits are the real envelope
    "head_x": 0.05, "head_y": 0.05, "head_z": 0.05,
    "head_yaw": np.pi / 2, "head_pitch": np.pi / 4, "head_roll": np.pi / 4,
}


# ---------------------------------------------------------------------------
def tf(xyz=(0, 0, 0), rpy=(0, 0, 0)) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def rotz(q: float) -> np.ndarray:
    return tf(rpy=(0, 0, q))


def fmt_origin(T: np.ndarray) -> str:
    xyz = T[:3, 3]
    rpy = R.from_matrix(T[:3, :3]).as_euler("xyz")
    xyz = [0.0 if abs(v) < 1e-9 else v for v in xyz]
    rpy = [0.0 if abs(v) < 1e-9 else v for v in rpy]
    return f'<origin xyz="{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}" rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"/>'


def mesh_name(visual) -> str:
    return visual.geometry.mesh.filename.split("/")[-1]


def rgba_of(visual):
    if visual.material is not None and visual.material.color is not None:
        return [float(x) for x in visual.material.color.rgba]
    return [0.7, 0.7, 0.7, 1.0]


def visual_xml(mesh: str, T: np.ndarray, rgba, indent="    ") -> str:
    name = "c_" + "".join(f"{int(round(c * 255)):02x}" for c in rgba)
    return (
        f"{indent}<visual>\n"
        f"{indent}  {fmt_origin(T)}\n"
        f'{indent}  <geometry><mesh filename="../meshes/{mesh}"/></geometry>\n'
        f'{indent}  <material name="{name}"><color rgba="{rgba[0]:.4f} {rgba[1]:.4f} {rgba[2]:.4f} {rgba[3]:.4f}"/></material>\n'
        f"{indent}</visual>\n"
    )


def link_visuals(link, T_pre: np.ndarray, used: set) -> str:
    out = ""
    for v in link.visuals:
        if v.geometry is None or v.geometry.mesh is None:
            continue
        m = mesh_name(v)
        if m.startswith(SKIP_MESH_PREFIXES):
            continue
        used.add(m)
        out += visual_xml(m, T_pre @ v.origin, rgba_of(v))
    return out


def min_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.dot(a, b))
    if s < 1e-12:
        return np.eye(3) if c > 0 else R.from_rotvec(np.pi * _any_perp(a)).as_matrix()
    return R.from_rotvec(v / s * np.arctan2(s, c)).as_matrix()


def _any_perp(a):
    p = np.cross(a, [1, 0, 0])
    if np.linalg.norm(p) < 1e-6:
        p = np.cross(a, [0, 1, 0])
    return p / np.linalg.norm(p)


def stewart_neutral_angles() -> list[float]:
    try:
        from reachy_mini.kinematics.analytical_kinematics import AnalyticalKinematics

        k = AnalyticalKinematics(automatic_body_yaw=False)
        assert abs(k.head_z_offset - HEAD_Z_OFFSET) < 1e-9, k.head_z_offset
        q = k.ik(np.eye(4), body_yaw=0.0)
        print("stewart angles at neutral from SDK IK (deg):", np.round(np.degrees(q[1:]), 3).tolist())
        return [float(x) for x in q[1:]]
    except Exception as exc:  # pragma: no cover - depends on the venv
        print(f"WARNING: reachy_mini SDK IK unavailable ({type(exc).__name__}: {exc}); using fallback angles")
        return list(STEWART_Q_NEUTRAL_FALLBACK)


# ---------------------------------------------------------------------------
def build(decimate: float, skip_meshes: bool) -> None:
    u = yourdfpy.URDF.load(VENDOR_URDF, load_meshes=False, build_scene_graph=True)
    u.update_cfg(np.zeros(len(u.actuated_joint_names)))
    W = u.base_link
    T_w = lambda frame: u.get_transform(frame, W)  # noqa: E731

    # --- frames -----------------------------------------------------------
    T_world_head0 = T_w("head")                       # vendor zero config (motors 0): head at z=0.1496
    T_xl_head = u.joint_map["head_frame"].origin      # xl_330 -> head (fixed)
    T_head_xl = np.linalg.inv(T_xl_head)
    T_world_body = u.joint_map["yaw_body"].origin     # foot -> body_down at yaw 0
    T_body_world = np.linalg.inv(T_world_body)
    T_world_head_neutral = tf(xyz=(0, 0, HEAD_Z_OFFSET))
    T_body_head_neutral = T_body_world @ T_world_head_neutral
    print("vendor zero-config head pose (world):", np.round(T_world_head0[:3, 3], 5).tolist(),
          "rpy", np.round(R.from_matrix(T_world_head0[:3, :3]).as_euler("xyz", degrees=True), 4).tolist())
    print("SDK neutral head pose (world): [0, 0, %.3f], identity" % HEAD_Z_OFFSET)

    # sanity: head frame is x-forward, y-left, z-up (camera at +x, right antenna at -y)
    cam = (np.linalg.inv(T_world_head0) @ T_w("camera"))[:3, 3]
    ant_r = (np.linalg.inv(T_world_head0) @ T_w("dc15_a01_horn_dummy_7"))[:3, 3]
    ant_l = (np.linalg.inv(T_world_head0) @ T_w("dc15_a01_horn_dummy_8"))[:3, 3]
    assert cam[0] > 0.02 and abs(cam[1]) < 1e-3, cam
    assert ant_r[1] < -0.03 < 0.03 < ant_l[1], (ant_r, ant_l)
    print("camera in head frame:", np.round(cam, 4).tolist(), " right antenna y=%.4f left antenna y=%.4f" % (ant_r[1], ant_l[1]))

    used: set = set()
    X = ['<?xml version="1.0"?>',
         "<!-- GENERATED by scripts/reachy_build_urdf.py from vendor/urdf/robot_no_collision.urdf -->",
         "<!-- Pollen Robotics reachy_mini description, Apache-2.0: https://github.com/pollen-robotics/reachy_mini -->",
         "<!-- Serial visualization chain; see README.md next to this file. Joint values are reachy_mini SDK units: -->",
         "<!--   head_x/y/z metres, head_roll/pitch/yaw radians (Rz*Ry*Rx, +pitch = nose down), antennas + body_yaw radians. -->",
         '<robot name="reachy_mini">']

    # --- base (foot) --------------------------------------------------------
    X.append('  <link name="base">')
    X.append(link_visuals(u.link_map["body_foot_3dprint"], np.eye(4), used).rstrip("\n"))
    X.append("  </link>")

    # --- body_yaw -----------------------------------------------------------
    jy = u.joint_map["yaw_body"]
    X.append('  <joint name="body_yaw" type="revolute">')
    X.append(f"    {fmt_origin(jy.origin)}")
    X.append('    <parent link="base"/>\n    <child link="body"/>\n    <axis xyz="0 0 1"/>')
    X.append(f'    <limit effort="10" velocity="8" lower="{jy.limit.lower:.6f}" upper="{jy.limit.upper:.6f}"/>')
    X.append("  </joint>")
    X.append('  <link name="body">')
    X.append(link_visuals(u.link_map["body_down_3dprint"], np.eye(4), used).rstrip("\n"))
    X.append("  </link>")

    # --- static Stewart assembly at the neutral ------------------------------
    q = stewart_neutral_angles()
    rod_len = float(np.linalg.norm(u.joint_map["closing_1_1_frame"].origin[:3, 3]))
    sx = ""
    max_len_err = 0.0
    for i in range(1, 7):
        sfx = "" if i == 1 else f"_{i}"
        horn = u.link_map[f"dc15_a01_horn_dummy{sfx}"]
        rod = u.link_map[f"stewart_link_rod{sfx}"]
        T_body_horn = u.joint_map[f"stewart_{i}"].origin @ rotz(q[i - 1])
        sx += link_visuals(horn, T_body_horn, used)
        # arm tip (passive_i_x origin) in body frame
        T_horn_tip = u.joint_map[f"passive_{i}_x"].origin
        P = (T_body_horn @ T_horn_tip)[:3, 3]
        # head-side attachment point, fixed in the head frame
        if i <= 5:
            Q_head = (np.linalg.inv(T_world_head0) @ T_w(f"closing_{i}_1"))[:3, 3]
            a = u.joint_map[f"closing_{i}_1_frame"].origin[:3, 3]      # attachment in rod frame
        else:
            Q_head = (np.linalg.inv(T_world_head0) @ T_w("passive_7_link_x"))[:3, 3]
            a = u.joint_map["passive_7_x"].origin[:3, 3]
        Q = (T_body_head_neutral @ np.append(Q_head, 1.0))[:3]
        d = Q - P
        max_len_err = max(max_len_err, abs(np.linalg.norm(d) - np.linalg.norm(a)))
        x_new = np.sign(a[0]) * d / np.linalg.norm(d)
        # vendor-zero rod pose (body frame) -> rotate minimally so its x axis follows the new direction
        T_body_rod0 = T_body_world @ T_w(f"stewart_link_rod{sfx}")
        Rn = min_rotation(T_body_rod0[:3, 0], x_new) @ T_body_rod0[:3, :3]
        T_body_rod = np.eye(4)
        T_body_rod[:3, :3] = Rn
        T_body_rod[:3, 3] = P
        sx += link_visuals(rod, T_body_rod, used)
    print(f"rod length {rod_len:.4f} m; max |attachment distance - rod length| at neutral = {max_len_err * 1000:.3f} mm")
    assert max_len_err < 1.5e-3, "Stewart rods do not close at the neutral pose - IK/frames inconsistent"
    X.append('  <joint name="stewart_static_fixed" type="fixed">')
    X.append('    <origin xyz="0 0 0" rpy="0 0 0"/>\n    <parent link="body"/>\n    <child link="stewart_static"/>')
    X.append("  </joint>")
    X.append('  <link name="stewart_static">')
    X.append(sx.rstrip("\n"))
    X.append("  </link>")

    # --- virtual 6-DoF head chain, attached to base at the SDK neutral -------
    chain = [
        ("head_x", "prismatic", "1 0 0", T_world_head_neutral, HEAD_LIMITS["head_x"], 1.0),
        ("head_y", "prismatic", "0 1 0", np.eye(4), HEAD_LIMITS["head_y"], 1.0),
        ("head_z", "prismatic", "0 0 1", np.eye(4), HEAD_LIMITS["head_z"], 1.0),
        ("head_yaw", "revolute", "0 0 1", np.eye(4), HEAD_LIMITS["head_yaw"], 8.0),
        ("head_pitch", "revolute", "0 1 0", np.eye(4), HEAD_LIMITS["head_pitch"], 8.0),
        ("head_roll", "revolute", "1 0 0", np.eye(4), HEAD_LIMITS["head_roll"], 8.0),
    ]
    parent = "base"
    for k, (name, jtype, axis, origin, lim, vel) in enumerate(chain):
        child = "head" if k == len(chain) - 1 else f"{name}_link"
        X.append(f'  <joint name="{name}" type="{jtype}">')
        X.append(f"    {fmt_origin(origin)}")
        X.append(f'    <parent link="{parent}"/>\n    <child link="{child}"/>\n    <axis xyz="{axis}"/>')
        X.append(f'    <limit effort="10" velocity="{vel}" lower="{-lim:.6f}" upper="{lim:.6f}"/>')
        X.append("  </joint>")
        if child != "head":
            X.append(f'  <link name="{child}"/>')
        parent = child

    X.append('  <link name="head">')
    X.append(link_visuals(u.link_map["xl_330"], T_head_xl, used).rstrip("\n"))
    X.append("  </link>")

    # --- antennas -----------------------------------------------------------
    for name, vj, vlink in (("antenna_right", "right_antenna", "dc15_a01_horn_dummy_7"),
                            ("antenna_left", "left_antenna", "dc15_a01_horn_dummy_8")):
        j = u.joint_map[vj]
        X.append(f'  <joint name="{name}" type="revolute">')
        X.append(f"    {fmt_origin(T_head_xl @ j.origin)}")
        # axis flipped: vendor model angle = -(real robot angle); see module docstring
        X.append(f'    <parent link="head"/>\n    <child link="{name}_link"/>\n    <axis xyz="0 0 -1"/>')
        X.append(f'    <limit effort="10" velocity="8" lower="{-np.pi:.6f}" upper="{np.pi:.6f}"/>')
        X.append("  </joint>")
        X.append(f'  <link name="{name}_link">')
        X.append(link_visuals(u.link_map[vlink], np.eye(4), used).rstrip("\n"))
        X.append("  </link>")

    X.append("</robot>")
    os.makedirs(os.path.dirname(OUT_URDF), exist_ok=True)
    with open(OUT_URDF, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(X) + "\n")
    print(f"wrote {OUT_URDF} ({os.path.getsize(OUT_URDF)} bytes, {len(used)} unique meshes)")

    # --- meshes ---------------------------------------------------------------
    if not skip_meshes:
        copy_meshes(sorted(used), decimate)

    # --- self-check with yourdfpy ------------------------------------------------
    v = yourdfpy.URDF.load(OUT_URDF, load_meshes=False, build_scene_graph=True)
    v.update_cfg(np.zeros(len(v.actuated_joint_names)))
    T = v.get_transform("head", v.base_link)
    print("self-check: joints", v.actuated_joint_names)
    print("self-check: head origin at zero =", np.round(T[:3, 3], 4).tolist(),
          "rpy", np.round(R.from_matrix(T[:3, :3]).as_euler("xyz"), 6).tolist())
    assert np.allclose(T[:3, 3], [0, 0, HEAD_Z_OFFSET]) and np.allclose(T[:3, :3], np.eye(3))


def copy_meshes(names: list[str], decimate: float) -> None:
    import trimesh

    os.makedirs(MESH_DIR, exist_ok=True)
    for f in os.listdir(MESH_DIR):
        if f.endswith(".stl"):
            os.remove(os.path.join(MESH_DIR, f))
    total = 0
    for n in names:
        src = os.path.join(VENDOR_ASSETS, n)
        dst = os.path.join(MESH_DIR, n)
        m = trimesh.load(src, force="mesh")
        nf = len(m.faces)
        if decimate < 1.0 and nf > 400:
            try:
                m = m.simplify_quadric_decimation(percent=decimate)
            except Exception as exc:  # pragma: no cover
                print(f"  decimation failed for {n}: {exc}; copying as-is")
                shutil.copyfile(src, dst)
                total += os.path.getsize(dst)
                continue
        m.export(dst, file_type="stl")
        total += os.path.getsize(dst)
        print(f"  {n:36s} {nf:6d} -> {len(m.faces):6d} faces, {os.path.getsize(dst) / 1024:7.1f} KB")
    print(f"meshes: {len(names)} files, {total / 1024 / 1024:.2f} MB (budget {MESH_BUDGET_BYTES / 1024 / 1024:.0f} MB)")
    if total > MESH_BUDGET_BYTES:
        print("ERROR: mesh folder over budget; lower --decimate", file=sys.stderr)
        sys.exit(1)
    with open(os.path.join(MESH_DIR, "ATTRIBUTION.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(
            "# Mesh attribution\n\n"
            "These STL files are Pollen Robotics' Reachy Mini description meshes\n"
            "(`reachy_mini/descriptions/reachy_mini/urdf/assets`), licensed **Apache-2.0**.\n\n"
            "Source: https://github.com/pollen-robotics/reachy_mini\n\n"
            f"They were copied from `../vendor/urdf/assets` and decimated to ~{int(decimate * 100)}% of their\n"
            "triangle count by `scripts/reachy_build_urdf.py` (trimesh + fast-simplification) to keep the\n"
            "folder under 8 MB for the browser viewer. Screws and connectors (`phs_*`, `bts2_*`, `b3b_eh*`)\n"
            "are omitted. The unmodified originals remain in `../vendor/`.\n"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decimate", type=float, default=0.5, help="fraction of triangles to keep (default 0.5)")
    ap.add_argument("--skip-meshes", action="store_true", help="only rewrite the URDF")
    a = ap.parse_args()
    build(a.decimate, a.skip_meshes)
