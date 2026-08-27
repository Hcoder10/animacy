# Mesh attribution

The STL files in this directory are derived from the Autonomous Lamp CAD
published by Autonomous in **Autonomous OS**:

- Source: https://github.com/autonomous-ai/autonomous-os/tree/main/robots/lamp/hardware/cad
  (per-part STLs in `stl/`, and `lamp.glb`, the full assembly export)
- Copyright: Autonomous (autonomous-ai)
- License: Apache-2.0 (the `robots/` tree of Autonomous OS)

What was changed (see `scripts/lamp_extract_meshes.py`):

- Surfaces come from the vendor's per-part STLs (watertight printed-part
  shells, millimetres). They are exported in local part frames, so each
  part-assembly was placed by rigid registration (PCA-initialised ICP) onto
  the matching solid of the assembled `lamp.glb`, whose armature also gives
  the joint pivots. Servos, PCBs and screws (present in the GLB, absent from
  the STLs) are not shown.
- Coordinates were converted to URDF (z up, x forward, metres) and each link
  mesh was re-based on its joint pivot. Parts above 9 000 faces were
  decimated (quadric, `fast_simplification`) to keep the folder under 6 MB.

| file | vendor parts | link |
|---|---|---|
| `base.stl` | `base`, `base-cap`, `button` | `base` |
| `swivel.stl` | `swivel-part-part1/2/3` | `swivel` |
| `lower_arm.stl` | `arm-1-part1/2` | `lower_arm` |
| `upper_arm.stl` | `arm-2-part1/2` | `upper_arm` |
| `neck.stl` | `neck`, `cap-servo` | `neck` |
| `head.stl` | `head-part1/2/3`, `light-cover` | `head` |

No files from LeLamp (GPL-3.0) were copied; the LeLamp URDF was only read for
comparison.
