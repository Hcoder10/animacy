"""Build and render the animacy demo film in one command.

    python scripts/video/edit_all.py

Re-runnable at any time. Whatever footage exists gets cut in; whatever is
missing becomes a labelled placeholder in the timeline and is listed in the
report, so this can be run the moment the first narration line lands and again
after every delivery.

Options:
    --no-render     build the EDL and the project file only
    --engine melt|ffmpeg|auto
    --max-runtime   seconds; lines marked droppable are cut to fit (default 210)
    --skip-derivatives
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from edit_common import (  # noqa: E402
    build_edl, save_edl, EDIT_DATA, EDIT_DOCS, MEDIA, find_melt, mmss, FFMPEG,
    MAX_RUNTIME, REPO,
)
import edit_assets  # noqa: E402
import edit_mlt  # noqa: E402
import edit_render as R  # noqa: E402

KDENLIVE = EDIT_DOCS / "animacy_demo.kdenlive"
RENDER_MLT = EDIT_DATA / "animacy_demo.mlt"
REPORT = EDIT_DATA / "report.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--engine", choices=["auto", "melt", "ffmpeg"], default="auto")
    ap.add_argument("--max-runtime", type=float, default=MAX_RUNTIME)
    ap.add_argument("--skip-derivatives", action="store_true")
    ap.add_argument("--qc", type=int, default=15)
    ap.add_argument("--suffix", default="",
                    help="render to animacy_demo<suffix>.mp4 instead of over the "
                         "current deliverables, e.g. --suffix _v2")
    args = ap.parse_args()

    if args.suffix:
        R.MASTER = MEDIA / f"animacy_demo{args.suffix}.mp4"
        R.MASTER_720 = MEDIA / f"animacy_demo_720{args.suffix}.mp4"
        R.LAMP_LOOP = MEDIA / f"animacy_lamp_loop{args.suffix}.mp4"

    if not FFMPEG:
        print("ffmpeg is required and was not found on PATH", file=sys.stderr)
        return 2

    t0 = time.time()
    edl = build_edl(max_runtime=args.max_runtime)
    log = edl.log

    edit_assets.build_all(edl, log)
    save_edl(edl)

    melt = find_melt()
    composite_for_render = edit_mlt.pick_composite(melt)

    # the project a human opens: Kdenlive's own compositing service
    edit_mlt.write_project(edl, KDENLIVE, composite="frei0r.cairoblend", log=log)
    # the one melt renders: whatever this melt build actually has
    edit_mlt.write_project(edl, RENDER_MLT, composite=composite_for_render)
    log.append(f"project: {KDENLIVE.relative_to(REPO)} "
               f"(render copy uses {composite_for_render})")

    engine_used = None
    if not args.no_render:
        mix = R.build_mix(edl, log)
        MEDIA.mkdir(parents=True, exist_ok=True)

        if args.engine in ("auto", "melt") and melt:
            if R.render_melt(RENDER_MLT, R.MASTER, log):
                engine_used = "melt (MLT)"
                R.fit_size(lambda vb: R.render_melt(RENDER_MLT, R.MASTER, log, bitrate=vb),
                           R.MASTER, R.MAX_MASTER_MB, log, "master")
        if engine_used is None and args.engine in ("auto", "ffmpeg"):
            if R.render_ffmpeg(edl, mix, R.MASTER, log):
                engine_used = "ffmpeg filter_complex"
                R.fit_size(lambda vb: R.render_ffmpeg(edl, mix, R.MASTER, log, bitrate=vb),
                           R.MASTER, R.MAX_MASTER_MB, log, "master")

        if engine_used is None:
            log.append("render: FAILED on every engine")
        else:
            log.append(f"render: master produced by {engine_used}")
            if not args.skip_derivatives:
                R.make_720(R.MASTER, R.MASTER_720, log)
                R.make_lamp_loop(edl, R.LAMP_LOOP, log)
            R.black_frame_check(R.MASTER, edl, log)
            R.qc_frames(R.MASTER, edl, log, n=args.qc)

    save_edl(edl)
    report = {
        "engine": engine_used,
        "runtime_s": round(edl.total, 2),
        "runtime": mmss(edl.total),
        "lines": len(edl.lines),
        "shots": len(edl.shots),
        "dissolves": len(edl.dissolves),
        "lower_thirds": [t.text for t in edl.titles],
        "dropped_lines": edl.dropped,
        "placeholders": [f"{s.label} s{s.section} @{s.start:.1f}s"
                         for s in edl.shots if s.kind == "slug"],
        "deliverables": {
            p.name: {"exists": p.exists(), "mb": round(R.mb(p), 2)}
            for p in (R.MASTER, R.MASTER_720, R.LAMP_LOOP, KDENLIVE)
        },
        "build_seconds": round(time.time() - t0, 1),
        "log": log,
    }
    EDIT_DATA.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n".join(log))
    print(f"\nruntime {mmss(edl.total)}  |  engine {engine_used}  |  "
          f"{report['build_seconds']}s")
    if report["placeholders"]:
        print(f"placeholders still in the cut: {len(report['placeholders'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
