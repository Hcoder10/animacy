"""URDF helpers shared by ``animacy preview`` and the tests.

* :func:`load_urdf` — yourdfpy load with mesh paths resolved relative to the
  URDF file (``../meshes/x.stl`` as every ``robots/<name>/urdf`` uses).
* :func:`pick_tip_link` / :func:`pick_gaze_axis` — the robot's "face": the link
  whose position and pointing direction the sign probe reports. Overridable
  from ``ROBOT.md`` (``description.viewer.tip_link`` and ``viewer.gaze`` =
  ``[x, y, z]`` in the tip link's frame) or the CLI; otherwise a link named
  ``head``/``gripper_frame_link``/... wins, else the link with the most actuated
  joints between it and the base. The gaze axis defaults to whichever of the
  tip link's ±x/±y/±z points most forward (base +x) at ``rest``.
* :func:`joints_from_channels` — canonical channel values → joint values
  through the profile's mapping (``raw_joint_targets``), one row.
* :func:`set_joints` — joint values (ROBOT.md units) → URDF config through
  ``to_urdf_values`` (``urdf_sign``/``urdf_offset``/units), applied to the model.
* :func:`probe` — for each joint, what +10 units does to the tip position
  (mm) and pointing direction (deg): the "sign table" a headless agent needs.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .profile import Profile
from .retarget import raw_joint_targets, to_urdf_values
from .schema import CHANNELS

TIP_PREFERENCE = ("head", "gripper_frame_link", "gripper_frame", "tcp", "tool0", "tool_link", "tool",
                  "ee_link", "end_effector", "tip", "hand", "gripper_link", "gripper")
AXES = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1), "-x": (-1, 0, 0), "-y": (0, -1, 0), "-z": (0, 0, -1)}
# "+10 units" per joint unit: what the probe adds to `rest`
PROBE_STEP = {"deg": 10.0, "mm": 10.0, "rad": math.radians(10.0), "m": 0.01, "unit": 0.1}


# ---------------------------------------------------------------- loading
def load_urdf(profile: Profile, load_meshes: bool = True, display_faces: Optional[int] = None):
    """yourdfpy model for a profile. ``display_faces`` caps triangles per geometry (for rendering)."""
    import yourdfpy

    path = profile.urdf_path()
    d = os.path.dirname(os.path.abspath(path))
    u = yourdfpy.URDF.load(path, load_meshes=load_meshes, build_scene_graph=True,
                           filename_handler=lambda fname: yourdfpy.filename_handler_relative(fname, dir=d))
    if load_meshes and display_faces:
        import trimesh

        for name, g in list(u.scene.geometry.items()):
            if isinstance(g, trimesh.Trimesh) and len(g.faces) > display_faces:
                try:
                    u.scene.geometry[name] = g.simplify_quadric_decimation(face_count=display_faces)
                except Exception:  # fast-simplification missing: draw the full mesh
                    pass
    return u


def child_joint_map(u) -> Dict[str, object]:
    """link name -> the joint whose child it is."""
    return {j.child: j for j in u.joint_map.values()}


def actuated_depth(u, link: str) -> int:
    """Number of actuated joints on the path base -> link."""
    cj = child_joint_map(u)
    act = set(u.actuated_joint_names)
    n = 0
    while link in cj:
        j = cj[link]
        if j.name in act:
            n += 1
        link = j.parent
    return n


def pick_tip_link(profile: Profile, u, override: Optional[str] = None) -> str:
    hint = override or (profile.description.viewer or {}).get("tip_link")
    if hint:
        if hint not in u.link_map:
            raise KeyError(f"tip link {hint!r} not in URDF (links: {sorted(u.link_map)})")
        return hint
    for name in TIP_PREFERENCE:
        if name in u.link_map:
            return name
    best = max(u.link_map, key=lambda l: (actuated_depth(u, l), -len(l)))
    return best


def link_of_node(u, node: str) -> str:
    """Scene-graph geometry node -> the URDF link it hangs from."""
    g = u.scene.graph
    n = node
    while n is not None and n not in u.link_map:
        n = g.transforms.parents.get(n)
    return n or node


# ---------------------------------------------------------------- FK through the profile
def rest_joints(profile: Profile) -> Dict[str, float]:
    return {j.name: float(j.rest) for j in profile.joints}


def joints_from_channels(profile: Profile, mode: str, channels: Dict[str, float]) -> Tuple[Dict[str, float], pd.DataFrame]:
    """One canonical frame (unspecified channels = 0 = neutral) -> joint values via the mapping."""
    row = {c: 0.0 for c in CHANNELS}
    row.update({k: float(v) for k, v in channels.items()})
    row["t"] = 0.0
    table = raw_joint_targets(pd.DataFrame({k: [v] for k, v in row.items()}), profile, mode)
    return {j: float(table[j].iloc[0]) for j in profile.joint_names}, table


def table_from_joints(profile: Profile, joints: Dict[str, float]) -> pd.DataFrame:
    row = {"t": [0.0]}
    for j in profile.joints:
        row[j.name] = [float(joints.get(j.name, j.rest))]
    return pd.DataFrame(row)


def set_joints(u, profile: Profile, joints: Dict[str, float]) -> Dict[str, float]:
    """Apply joint values (ROBOT.md units) to the model; returns the URDF config (rad / m)."""
    vals = to_urdf_values(table_from_joints(profile, joints), profile)
    cfg = {k: float(v[0]) for k, v in vals.items()}
    u.update_cfg(cfg)
    return cfg


def set_table_row(u, profile: Profile, table: pd.DataFrame, i: int) -> Dict[str, float]:
    joints = {j: float(table[j].iloc[i]) for j in profile.joint_names if j in table.columns}
    set_joints(u, profile, joints)
    return joints


def link_pose(u, link: str) -> np.ndarray:
    return u.get_transform(link, u.base_link)


def pick_gaze_axis(profile: Profile, u, tip: str, override: Optional[str] = None) -> Tuple[np.ndarray, str]:
    """Unit vector in the tip link's frame that is its 'pointing' direction, + a label."""
    hint = override or (profile.description.viewer or {}).get("gaze")
    if hint is not None:
        if isinstance(hint, str) and hint.strip() in AXES:
            v = np.asarray(AXES[hint.strip()], dtype=float)
            return v, hint.strip()
        vals = [float(x) for x in (hint.split(",") if isinstance(hint, str) else hint)]
        v = np.asarray(vals, dtype=float)
        n = np.linalg.norm(v)
        if len(vals) != 3 or n < 1e-9:
            raise ValueError(f"gaze must be x|y|z|-x|-y|-z or three numbers, got {hint!r}")
        return v / n, "custom(%.2f,%.2f,%.2f)" % tuple(v / n)
    set_joints(u, profile, rest_joints(profile))
    R = link_pose(u, tip)[:3, :3]
    best = max(AXES, key=lambda k: float((R @ np.asarray(AXES[k], dtype=float))[0]))
    return np.asarray(AXES[best], dtype=float), best


def tip_state(u, tip: str, gaze: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    T = link_pose(u, tip)
    return T[:3, 3].copy(), T[:3, :3] @ gaze


def yaw_pitch_deg(d: np.ndarray) -> Tuple[float, float]:
    d = d / (np.linalg.norm(d) + 1e-12)
    return math.degrees(math.atan2(d[1], d[0])), math.degrees(math.asin(max(-1.0, min(1.0, d[2]))))


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------- probe
def _reads_as(dx, dy, dz, dyaw, dpitch) -> str:
    effects = []
    if abs(dyaw) > 1.0:
        effects.append((abs(dyaw) / 10.0, f"gaze {'LEFT' if dyaw > 0 else 'RIGHT'} {abs(dyaw):.1f} deg"))
    if abs(dpitch) > 1.0:
        effects.append((abs(dpitch) / 10.0, f"gaze {'UP' if dpitch > 0 else 'DOWN'} {abs(dpitch):.1f} deg"))
    for v, pos, neg in ((dx, "forward", "back"), (dy, "left", "right"), (dz, "up", "down")):
        if abs(v) > 3.0:
            effects.append((abs(v) / 30.0, f"tip {pos if v > 0 else neg} {abs(v):.0f} mm"))
    if not effects:
        return "no effect on the tip"
    effects.sort(key=lambda e: -e[0])
    return ", ".join(e[1] for e in effects[:2])


def probe(profile: Profile, u, tip: str, gaze: np.ndarray) -> List[Dict]:
    """Per joint: +10 units from `rest` -> tip displacement (mm) and gaze yaw/pitch change (deg).

    Joints that do not move the tip (e.g. antennas downstream of the head) are
    probed on their own child link instead: a point 5 cm up that link's z axis.
    """
    cj = child_joint_map(u)
    rest = rest_joints(profile)
    set_joints(u, profile, rest)
    p0, g0 = tip_state(u, tip, gaze)
    yaw0, pitch0 = yaw_pitch_deg(g0)
    rows = []
    for j in profile.joints:
        step = PROBE_STEP.get(j.unit, 10.0)
        if j.rest + step > j.max and j.rest - step >= j.min:
            step = -step
        q = dict(rest)
        q[j.name] = j.rest + step
        set_joints(u, profile, q)
        p, g = tip_state(u, tip, gaze)
        yaw, pitch = yaw_pitch_deg(g)
        d = (p - p0) * 1000.0
        dyaw, dpitch = _wrap(yaw - yaw0), pitch - pitch0
        row = {"joint": j.name, "unit": j.unit, "step": step, "rest": j.rest, "urdf_sign": j.urdf_sign,
               "dx_mm": float(d[0]), "dy_mm": float(d[1]), "dz_mm": float(d[2]),
               "dyaw_deg": float(dyaw), "dpitch_deg": float(dpitch), "probe_link": tip}
        if np.linalg.norm(d) < 0.5 and abs(dyaw) < 0.1 and abs(dpitch) < 0.1:
            # tip unaffected (joint is parallel to / downstream of the tip): probe the joint's own
            # child link instead, at a point 5 cm forward and 5 cm up (world axes) from its origin
            # so it is off any plausible hinge axis
            uj = u.joint_map.get(j.urdf_joint)
            if uj is not None and uj.child in u.link_map:
                set_joints(u, profile, rest)
                T0 = link_pose(u, uj.child)
                pt = np.append(T0[:3, 3] + np.array([0.05, 0.0, 0.05]), 1.0)
                pl = np.linalg.inv(T0) @ pt
                set_joints(u, profile, q)
                pt1 = link_pose(u, uj.child) @ pl
                dd = (pt1[:3] - pt[:3]) * 1000.0
                row.update({"dx_mm": float(dd[0]), "dy_mm": float(dd[1]), "dz_mm": float(dd[2]),
                            "probe_link": uj.child + " (+5cm fwd,+5cm up)"})
        row["reads_as"] = _reads_as(row["dx_mm"], row["dy_mm"], row["dz_mm"], row["dyaw_deg"], row["dpitch_deg"])
        if row["probe_link"] != tip:
            row["reads_as"] = f"tip unaffected; {row['probe_link']}: " + row["reads_as"].replace("tip ", "")
        rows.append(row)
    set_joints(u, profile, rest)
    return rows


def format_probe(profile: Profile, rows: Sequence[Dict], tip: str, gaze_label: str) -> str:
    rest = ", ".join(f"{j.name}={j.rest:g}{'' if j.unit == 'unit' else j.unit}" for j in profile.joints)
    lines = [f"sign probe for {profile.name}: tip link `{tip}`, gaze axis {gaze_label} (base frame: x forward, y left, z up)",
             f"rest: {rest}",
             "each row: joint at rest + step (ROBOT.md units, through urdf_sign/offset) -> tip displacement, gaze change",
             f"{'joint':16s} {'step':>9s} {'tip dx':>7s} {'dy':>7s} {'dz (mm)':>8s} {'gaze dyaw':>10s} {'dpitch (deg)':>13s}  reads as"]
    for r in rows:
        step = f"{r['step']:+g}{r['unit'] if r['unit'] != 'unit' else ''}"
        lines.append(f"{r['joint']:16s} {step:>9s} {r['dx_mm']:7.1f} {r['dy_mm']:7.1f} {r['dz_mm']:8.1f} "
                     f"{r['dyaw_deg']:10.1f} {r['dpitch_deg']:13.1f}  {r['reads_as']}")
    return "\n".join(lines)


# ---------------------------------------------------------------- geometry for rendering
def scene_triangles(u) -> List[Tuple[np.ndarray, str]]:
    """[(faces Nx3x3 in the base frame, link name)] for the model's current config."""
    import trimesh

    out = []
    g = u.scene.graph
    for node in g.nodes_geometry:
        T, gname = g[node]
        geom = u.scene.geometry[gname]
        if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
            continue
        v = trimesh.transform_points(geom.vertices, T)
        out.append((v[geom.faces], link_of_node(u, node)))
    return out
