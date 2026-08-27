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
5. **Run `animacy check robots/<name>`** until it passes (it also refuses a
   profile range wider than the URDF's limits — that means a wrong sign/offset).
6. **Find your signs headlessly.** `animacy preview robots/<name>` renders PNGs
   of the rest pose and of each canonical calibration pose (look left, look up,
   roll, brows, lean in, puppet wave) *through your mapping*, and prints what
   +10 units on each joint does to the head/tip position. Read the PNGs (a
   coding agent can) before touching the browser. Until that command lands,
   `robots/so101/dev/render_previews.py` is the worked example.
7. **Preview in the browser.** `animacy profile export robots/<name>` then open
   `web/` and play a captured clip. Fix directions with `gain: -1`. Play the
   vendor's native clips (if any) first — if *those* look wrong, the URDF axes
   are wrong, not the mapping.
8. **Write the prose**: which signs are verified (sim vs hardware), what `rest`
   looks like, how frames reach the real robot, and the verification commands.
9. Add a row to the README robots table and open a PR. CI runs `animacy check`
   on every robot.

Worked example with timings: `robots/so101/ADDING_LOG.md` — the SO-101 arm was
added by a coding agent following this page in 26 minutes; its gap list drove
rules 7–10 of the spec.

## Rules of thumb

- Never edit captured data to fix a sign; fix it in `ROBOT.md`.
- Never raise `max_speed` above the vendor's safety ceiling because the motor
  can go faster. Time is stretched, not clipped, so nothing is lost.
- Joint names are an ABI once merged.
- If you had to write code to add the robot, something is missing from the
  spec — open an issue so the next robot needs only the `.md`.
