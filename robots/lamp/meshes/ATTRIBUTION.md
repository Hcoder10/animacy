# Mesh attribution

The STL files in this directory are derived from the Autonomous Lamp CAD
published by Autonomous in **Autonomous OS**:

- Source: https://github.com/autonomous-ai/autonomous-os/tree/main/robots/lamp/hardware/cad
  (`lamp.glb`, the full assembly export; per-part STLs in `stl/`)
- Copyright: Autonomous (autonomous-ai)
- License: Apache-2.0 (the `robots/` tree of Autonomous OS)

What was changed (see `scripts/lamp_extract_meshes.py`):

- `lamp.glb` was split by its own mesh groups (`0_base`, `1_base_yaw`,
  `2_base_pitch`, `3_elbow_pitch`, `4_wrist_roll`, `5_wrist_pitch`) into one
  STL per URDF link; the cable (`6_wire`) is not exported.
- Coordinates were changed from glTF (y up) to URDF (z up, x forward) and each
  link mesh was re-based on its joint pivot; units are metres.
- Meshes were decimated (quadric, `fast_simplification`) to keep the folder
  under 6 MB; the base shell is kept at full resolution and its internal
  electronics are dropped.

| file | GLB group | link |
|---|---|---|
| `base.stl` | `0_base` | `base` |
| `swivel.stl` | `1_base_yaw` | `swivel` |
| `lower_arm.stl` | `2_base_pitch` | `lower_arm` |
| `upper_arm.stl` | `3_elbow_pitch` | `upper_arm` |
| `neck.stl` | `4_wrist_roll` | `neck` |
| `head.stl` | `5_wrist_pitch` | `head` |

No files from LeLamp (GPL-3.0) were copied; the LeLamp URDF was only read for
comparison.
