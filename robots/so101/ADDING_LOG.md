# Adding the SO-101 by following `docs/ADD_A_ROBOT.md` literally — log

Date 2026-08-26, one agent session, wall clock from the first `date` to the last
green run. Goal: prove that a robot is a one-folder job and record where the
docs fell short. Everything lives in `robots/so101/` except `tests/test_so101.py`
and the two viewer outputs the doc asks for (`web/robots/so101.json`,
`web/manifest.json`).

## Wall time per step

| step (ADD_A_ROBOT.md numbering) | wall | notes |
|---|---|---|
| read ADD_A_ROBOT.md, template, ROBOT_MD_SPEC, lamp/reachy profiles | 22:35–22:38 (3 min) | |
| find + fetch a licensed URDF/meshes | 22:38–22:44 (6 min) | GitHub API listing took ~60 s; raw files were fast. Local copy of the same 13 STLs existed (g4studio) — re-downloaded from the Apache-2.0 repo for provenance (byte-identical sizes). |
| 1–2. scaffold folder, drop URDF in (mesh paths → `../meshes/`) | 22:44–22:45 (1 min) | `sed` only; URDF otherwise unmodified |
| FK exploration: frame axes, joint directions, rest-pose search | 22:45–22:50 (5 min) | needed *code* (yourdfpy) — the doc has no "find your signs" step |
| 3–4. write ROBOT.md (joints from URDF limits, `default` + `puppet` mappings, prose) | 22:50–22:53 (3 min) | |
| meshes: decimation script + ATTRIBUTION.md | 22:53–22:58 (5 min incl. one wrong-API retry) | trimesh's `percent` = fraction *removed*; 30% → 10.8 MB, so used `face_count` @ 15% → 4.5 MB |
| 5. `animacy check` | 22:53 (5 s) | passed first time |
| 6. preview (no browser here): matplotlib renderer through the mapping | 22:53–22:59 (6 min incl. fixes) | look-left/up/lean/wave all read correctly; found the puppet elbow sign needed `-1` |
| tests + `profile export` + `build_manifest` | 22:56–23:01 (5 min) | first run: limits rounded *outward* by float error, rest-y tolerance |
| **total** | **26 min** (22:35 → 23:01) to green: check, 6 tests, 5 previews, viewer JSON + manifest | plus this log and the final preview framing fix |

## Doc / spec gaps found (in order of pain)

1. **No "find your signs" step.** ADD_A_ROBOT says "fix directions with
   `gain: -1`" after eyeballing the browser — but a headless session cannot
   open `web/`, and there is no CLI that reports what a joint does. I had to
   write yourdfpy FK by hand to learn that +shoulder_pan swings right and
   +lift/+elbow/+wrist_flex all move the tip down. Suggest: `animacy preview
   --png` (offscreen matplotlib, like `robots/so101/dev/render_previews.py`)
   and/or `animacy urdf probe <robot>` printing, per joint, where the
   end-link moves for +10°.
2. **README had no robots table** when I checked (22:36). Step 8 says "add a
   row to the README table"; `README.md` only mentioned the two reference
   robots in prose. A table (with an `so101` row) appeared in the README from
   the lead's side before I finished, so no row was added from here.
3. **`rest` guidance assumes a vendor move library** ("median of their idle
   clip"). LeRobot ships task datasets, not expressive clips, and the arm's
   folded storage pose is not a conversational rest. Suggest a sentence for
   arms: "rest = the pose the body should *talk* from; search it with FK".
4. **`max_speed` guidance assumes a vendor safety file.** LeRobot has none
   (only `max_relative_target`). Wrote a conservative 120–200 deg/s and said
   so; the spec should name a default policy for that case.
5. **Units for LeRobot bodies are ambiguous.** LeRobot drives the bus in
   `RANGE_M100_100` unless `use_degrees=True`; the lamp profile documents the
   same trap. The spec should say which one an `export.formats: [lerobot]`
   profile promises (this one: degrees).
6. **Joint limits vs the URDF check.** Copying the URDF's radian limits
   converted to degrees put ±110/±95/−10 a float-epsilon *outside* the URDF's
   range (my test caught it, `animacy check` does not compare limits). A
   `check` rule "profile limits inside URDF limits" would be cheap.
7. **Mesh budget tooling.** The doc says "keep < 8 MB" but gives no decimation
   recipe; every robot so far grew its own script (`lamp_extract_meshes.py`,
   `reachy_build_urdf.py`, `so101/dev/build_meshes.py`). One shared
   `animacy meshes shrink <dir> --budget 6MB` would remove a script per robot.
8. **`offset` semantics for `puppet` chains.** `joint = rest + offset + gain·ch`
   means every 1:1 mapping needs `offset: -rest` (plus the anatomical offset).
   Works, but the doc example does not show it; an `absolute: true` flag (ignore
   `rest`) would make arm puppeteering a one-liner.
9. **The template's export section lists `[csv]`** while the real exporters are
   `autonomous_os_csv | pollen_move | lerobot | csv | json`; a comment listing
   them would save a lookup.

## What was verified and what was not

- Verified (sim, `tests/test_so101.py` + `urdf/preview/*.png`): profile checks
  clean; joint names/limits match the URDF; rest pose is upright and level; look
  left swings the gripper to +y, look up tips it up, lean-in reaches forward,
  mouth opens the gripper, brows tip the wrist; puppet chain is 1:1 with the
  documented signs; meshes 4.5 MB, all referenced files present.
- Not verified: anything on a physical SO-101 (no unit here). `wrist_roll`
  sign, LeRobot per-unit calibration sign flips, and `max_speed` are guesses
  flagged in `ROBOT.md`.
