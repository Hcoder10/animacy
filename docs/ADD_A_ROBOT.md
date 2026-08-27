# Add a robot (for people, and for Claude Code / Codex sessions)

Goal: after this, `animacy retarget --robot <name>` turns any canonical human
clip into your robot's motion, the browser viewer shows it on your URDF, and
the shared motion model drives it live — **without retraining anything**.

You produce exactly one folder:

```
robots/<name>/
  ROBOT.md          # the contract (front matter) + notes (prose)
  urdf/<name>.urdf  # visual/kinematic description; meshes next to it
  meshes/*.stl      # optional, referenced from the URDF (keep < 8 MB total)
  clips/native/     # optional: the vendor's own hand-authored moves
```

## Steps

1. **Copy the template.** `cp -r robots/_template robots/<name>` and rename.
2. **Put the URDF in.** Any URDF works for visualization. If the robot is a
   parallel mechanism (Stewart head, delta), write a *serial visualization
   chain* whose joints are the robot's *control* variables (e.g. `head_x/y/z`,
   `head_roll/pitch/yaw`) — see `robots/reachy_mini/urdf/README.md`.
3. **Fill the joint table** from the vendor's spec: names (use the vendor's
   control names so exports are 1:1), units, limits, rest pose, `max_speed`
   from the vendor's *safety* file if there is one. If the vendor ships
   recorded moves, take `rest` = median of their idle clip and keep limits
   inside the p1..p99 envelope of their library (`scripts/clip_envelope.py`).
4. **Write the `default` mapping by function, not anatomy.** Ask, for each
   canonical channel in `docs/CANONICAL.md`: *what on this body does the job?*
   - gaze (`head_yaw`, `head_pitch`) → whatever points the face.
   - lean (`torso_lean_fwd`, `head_x`) → base joints.
   - height (`head_z`) → whatever raises the head.
   - affect (`brow_l/r`, `mouth_open`) → the body's most legible expressive channel.
   - `speaking` is not mapped; it is a training-time role flag.
   Seed gains so a typical human move lands inside the vendor's envelope.
5. **Run `animacy check robots/<name>`** until it passes.
6. **Preview.** `animacy profile export robots/<name>` then open `web/` and
   play a captured clip. Fix directions with `gain: -1`. Play the vendor's
   native clips (if any) first — if *those* look wrong, the URDF axes are wrong,
   not the mapping.
7. **Write the prose**: which signs are verified (sim vs hardware), what `rest`
   looks like, how frames reach the real robot, and the verification commands.
8. Add a row to the README table and open a PR. CI runs `animacy check` on
   every robot.

## Rules of thumb

- Never edit captured data to fix a sign; fix it in `ROBOT.md`.
- Never raise `max_speed` above the vendor's safety ceiling because the motor
  can go faster. Time is stretched, not clipped, so nothing is lost.
- Joint names are an ABI once merged.
- If you had to write code to add the robot, something is missing from the
  spec — open an issue so the next robot needs only the `.md`.
