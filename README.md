# animacy

**The open interaction layer for expressive robots.** Human motion in, any
robot's motion out — one `ROBOT.md` per body, no retraining.

Desk robots feel alive when their motion has the statistics of living motion:
anticipation, hesitation, overshoot-and-settle, micro-movement, reactions timed
to speech. Today every body ships a menu of hand-authored clips (Autonomous
Lamp: 31, Reachy Mini: 85). animacy replaces the menu with a pipeline:

```
webcam / phone / licensed video ──capture──▶ canonical human motion (30 Hz)
                                                  │
                     ROBOT.md (lamp) ◀────────────┼───────────▶ ROBOT.md (reachy_mini) ◀─── ROBOT.md (yours)
                                                  │
        motion model: speech + text ──▶ canonical motion, streamed live while the robot talks
```

- `docs/CANONICAL.md` — the one motion space everything speaks.
- `docs/ROBOT_MD_SPEC.md` — the one file that adds a robot.
- `docs/ADD_A_ROBOT.md` — hand this to a Claude Code / Codex session.
- `robots/lamp`, `robots/reachy_mini` — reference bodies, both official
  Autonomous OS robots; exports are byte-compatible with their runtimes.
- `web/` — browser viewer: both URDFs side by side, vendor clips, captured
  clips, live webcam puppeteering, and the model.

Status: **under construction for the Autonomous Open Source Grant (Aug 28, 2026).**
Lineage: [reachy-duplex](https://github.com/Hcoder10/reachy-duplex) (full-duplex
speech + learned motion on a physical Reachy Mini).

License: Apache-2.0 for everything in this repo except where a folder carries
its own license file (`third_party/`).
