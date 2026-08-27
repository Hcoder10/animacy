"""Blind motion grading: the acceptance gate for animacy.

An outside judge (Kimi K3 through the local ``kimi`` CLI) watches reels of
short robot clips and scores each one. It is never told how a clip was made:
the clip -> origin map is sealed in a manifest that never enters the judge's
workspace, and the rubric carries no project vocabulary.

Modules:
  kimi       the CLI adapter (prompt via file, JSON extraction, timeouts)
  render     joint table -> MP4 through the three.js viewer (Playwright + ffmpeg)
  movements  the five movements, candidate + calibration clip construction
  reel       blind numbering, seeded shuffle, reel assembly, sealed manifest
  rubric     the judge's prompt (project-context free) + forbidden-word check
  run        the end-to-end run, unsealing, the pass rule, the reports

The pass rule (``run.gate``) is owned here and must not be weakened by any
other module: for each robot, the ``model`` source must reach overall >= 8.0
on ALL five movements, using the MEAN over seeds (best-of-seeds is reported
but never used for the verdict).
"""
