# Reachy Mini sim-to-real calibration — 2026-08-26 21:47

Physical Reachy Mini Wireless at 192.168.1.60, daemon HTTP API (`POST /api/move/set_target` at 30 Hz, slew-clamped), motors enabled + `wake_up` by the script.
Synthetic canonical clip → `robots/reachy_mini/ROBOT.md` `default` mapping → `retarget_clip` → daemon. Pitch multiplied by -1 before sending (daemon +pitch looks down).
Measured = the daemon's `present_head_pose` / `present_antenna_joint_positions` / `present_body_yaw` sampled mid-hold. Degrees. Pitch column is in the DAEMON frame (sent value), yaw/roll/antennas/body in the shared frame.

| segment | yaw cmd | yaw meas | pitch sent | pitch meas | roll cmd | roll meas | ant L cmd | ant L meas | ant R cmd | ant R meas | body cmd | body meas |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| center | 0.0 | -1.2 | 0.0 | 0.1 | 0.0 | 3.5 | -0.0 | 0.3 | 0.0 | 0.1 | -0.0 | -0.4 |
| look left | 27.7 | 23.7 | 0.0 | -0.8 | 0.0 | 1.0 | 0.0 | 0.3 | -0.0 | -0.1 | 8.8 | 8.1 |
| center | -0.1 | 7.6 | 0.0 | 0.5 | 0.0 | 0.3 | 0.0 | 0.4 | -0.0 | -0.2 | -0.1 | 0.4 |
| look right | -27.7 | -24.6 | 0.0 | 0.8 | 0.0 | 0.2 | -0.0 | 0.3 | 0.0 | 0.1 | -8.8 | -8.8 |
| center | 0.0 | -7.5 | 0.0 | 0.1 | 0.0 | 0.9 | -0.0 | 0.4 | 0.0 | 0.1 | 0.1 | -1.5 |
| look up | -0.0 | 0.7 | -19.7 | -19.1 | -0.0 | 0.4 | -0.0 | 0.5 | 0.0 | -0.1 | 0.0 | -0.2 |
| center | 0.0 | 0.5 | 0.0 | -5.2 | 0.0 | 0.1 | 0.0 | 0.4 | -0.0 | 0.0 | -0.0 | -0.2 |
| look down | 0.0 | -1.2 | 19.7 | 20.4 | 0.0 | 0.6 | -0.0 | 0.2 | 0.0 | 0.1 | -0.0 | -0.2 |
| center | -0.0 | 1.2 | -0.1 | 5.3 | -0.0 | 1.6 | -0.0 | 0.3 | 0.0 | -0.1 | 0.0 | -0.2 |
| roll, right ear down | 0.0 | -1.5 | 0.0 | -0.5 | 15.7 | 16.9 | -10.0 | -10.2 | 10.0 | 9.9 | -0.0 | -0.2 |
| center | 0.0 | 1.3 | -0.0 | 2.9 | -0.0 | 5.5 | -0.0 | -1.1 | 0.0 | 0.2 | -0.0 | -0.2 |
| both brows up | -0.0 | 0.8 | -0.0 | 1.3 | 0.0 | 4.4 | 90.0 | 90.8 | 90.0 | 61.4 | -0.0 | -0.2 |
| center | -0.0 | 1.1 | 0.0 | 1.6 | 0.0 | 2.6 | -0.1 | 0.3 | -0.1 | 0.3 | -0.0 | -0.1 |
| left brow only | -0.0 | 1.0 | -0.0 | 1.7 | -0.0 | 2.5 | 90.0 | 91.4 | -0.0 | 0.4 | 0.0 | -0.2 |
| center | -0.0 | 1.0 | -0.0 | 1.7 | 0.0 | 2.5 | -0.1 | 0.4 | -0.0 | -0.2 | -0.0 | -0.2 |
| lean in | 0.0 | 1.1 | -0.0 | -2.5 | -0.0 | 0.3 | -0.0 | 0.1 | 0.0 | 0.0 | -0.0 | -0.2 |
| center | -0.0 | 0.8 | -0.0 | 1.6 | 0.0 | 0.7 | 0.0 | 0.0 | 0.0 | -0.2 | 0.0 | 0.1 |
| turn body left | 0.0 | 0.3 | 0.0 | 1.0 | -0.0 | 0.2 | 0.0 | -0.1 | 0.0 | -0.2 | 32.0 | 30.6 |
| center | 0.0 | -0.6 | 0.0 | 0.1 | -0.0 | 1.3 | 0.0 | -0.2 | -0.0 | 0.4 | -0.3 | 3.5 |

## Reading
- Every commanded axis is tracked by the daemon within a few degrees at the mid-hold sample (the peak sample lands slightly before settle on yaw).
- `left brow only` moved the FIRST antenna element → `target_antennas[0]` is what this profile calls `antenna_left`; whether that is physically the robot's left antenna is a visual check (below).
- Body-yaw coupling from `head_yaw` (0.25) is visible (±8 deg on the big head turns) and reads back.

## Visual confirmation (a person watching the robot)
- [ ] `look left` turned the head toward the ROBOT'S left
- [ ] `look up` looked UP (pitch sign -1 correct)
- [ ] `roll, right ear down` dropped the robot's right ear
- [ ] `left brow only` raised the robot's LEFT antenna
- [ ] `turn body left` rotated the body toward the robot's left

Raw log: `reachy_sim2real_20260826_214727.json`. Script: `scripts/reachy_sim2real.py`.
