# Mesh attribution

These STL files are Pollen Robotics' Reachy Mini description meshes
(`reachy_mini/descriptions/reachy_mini/urdf/assets`), licensed **Apache-2.0**.

Source: https://github.com/pollen-robotics/reachy_mini

They were copied from `../vendor/urdf/assets` and decimated to ~50% of their
triangle count by `scripts/reachy_build_urdf.py` (trimesh + fast-simplification) to keep the
folder under 8 MB for the browser viewer. Screws and connectors (`phs_*`, `bts2_*`, `b3b_eh*`)
are omitted. The unmodified originals remain in `../vendor/`.
