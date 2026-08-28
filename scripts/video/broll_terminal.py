"""Terminal + document B-roll: real commands, real stdout, real files.

    python scripts/video/broll_terminal.py --shots check robotmd retarget

Every block on screen is stdout that this script actually captured from that
exact command, on this machine, in this repo. Nothing is typed by hand into a
mock-up; `web/dev/broll/term.html` only controls the *pace* at which the
captured bytes appear.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from broll_common import (  # noqa: E402
    ANIMACY, OUT_DIR, PY, ROOT, Block, TermShot, log, record_doc, record_term,
    run_bash, run_capture,
)

ARTIFACTS = os.path.join(ROOT, "out")


def pace(b: Block, *, cap: float = 2.6, floor: float = 0.3) -> Block:
    """Let the terminal wait roughly as long as the real command did."""
    if b.seconds is not None:
        b.output_delay = max(floor, min(cap, b.seconds))
    return b


# ---------------------------------------------------------------------------
# 3 — one file per robot
# ---------------------------------------------------------------------------
def shot_check() -> dict:
    shot = TermShot(title="animacy — validate every robot profile")
    shot.blocks = [
        run_bash("ls robots/", pause_after=0.55),
        pace(run_capture(f'"{ANIMACY}" check robots/lamp', display="animacy check robots/lamp", pause_after=0.5), cap=1.3),
        pace(run_capture(f'"{ANIMACY}" check robots/reachy_mini', display="animacy check robots/reachy_mini", pause_after=0.5), cap=1.3),
        pace(run_capture(f'"{ANIMACY}" check robots/so101', display="animacy check robots/so101", pause_after=0.9), cap=1.3),
    ]
    return record_term(shot, "s3_check_three_robots.mp4", section="3",
                       shows="`animacy check` validating all three shipped ROBOT.md profiles "
                             "(lamp, reachy_mini, so101) — joints, modes and URDF resolved for each.",
                       source="animacy check robots/{lamp,reachy_mini,so101}")


def _robot_md() -> list[str]:
    path = os.path.join(ROOT, "robots", "lamp", "ROBOT.md")
    with open(path, encoding="utf-8") as fh:
        return fh.read().replace("\r\n", "\n").split("\n")


def shot_robotmd() -> dict:
    """Front matter, description and the joint table with limits + max_speed."""
    lines = _robot_md()
    end = next(i for i, l in enumerate(lines) if l.startswith("retarget:"))
    log(f"  robots/lamp/ROBOT.md: {len(lines)} lines, showing 1..{end}")
    return record_doc("\n".join(lines[:end]), "s3_robotmd_lamp.mp4",
                      path_label="robots/lamp/ROBOT.md",
                      meta=f"{len(lines)} lines · schema animacy.robot.v1", px_per_sec=74,
                      section="3",
                      shows="Slow scroll over the head of the lamp's ROBOT.md: schema, vendor, licence, "
                            "rate_hz, the URDF it points at, the vendor's own servo notes and safety "
                            "ceiling, and the five joints with min / max / rest / max_speed.",
                      source=f"robots/lamp/ROBOT.md lines 1-{end}", max_seconds=15.0)


def shot_robotmd_mapping() -> dict:
    """The retarget block: human channels mixed onto joints, with the fitted gains."""
    lines = _robot_md()
    start = next(i for i, l in enumerate(lines) if l.startswith("  # Vendor signs"))
    end = next(i for i, l in enumerate(lines) if l.startswith("    wrist_pitch:"))
    log(f"  robots/lamp/ROBOT.md: showing the `default` mapping, lines {start + 1}..{end}")
    return record_doc("\n".join(lines[start:end]), "s4_robotmd_mapping.mp4",
                      path_label="robots/lamp/ROBOT.md  —  retarget: default", px_per_sec=64,
                      meta="canonical human channels -> lamp joints", section="4",
                      shows="The retarget mapping itself: base_yaw mixed from head_yaw and torso_yaw "
                            "with gains fitted from the vendor's own 31 clips, the negative signs that "
                            "encode the vendor's axis directions, deadbands and spring hz/zeta.",
                      source=f"robots/lamp/ROBOT.md lines {start + 1}-{end}", max_seconds=15.0)


# ---------------------------------------------------------------------------
# 4 — retarget
# ---------------------------------------------------------------------------
def shot_retarget() -> dict:
    os.makedirs(ARTIFACTS, exist_ok=True)
    out_csv = "out/lamp_obama.csv"
    cmd = ("animacy retarget --robot robots/lamp --clip data/clips/obama_2015_02_07 "
           f"-o {out_csv} --format autonomous_os_csv")
    real = cmd.replace("animacy", f'"{ANIMACY}"', 1) + " --force"
    shot = TermShot(title="animacy — canonical clip → Autonomous OS CSV")
    shot.type_cps = 25.0
    shot.blocks = [
        pace(run_capture(real, display=cmd, pause_after=0.5), cap=1.6),
        run_bash(f"head -n 4 {out_csv}", pause_after=0.6),
        run_bash(f"wc -l {out_csv}", pause_after=0.9),
    ]
    return record_term(shot, "s4_retarget_csv.mp4", section="4",
                       shows="A captured human clip retargeted onto the lamp and written in the exact "
                             "CSV Autonomous OS already accepts — the real header "
                             "(timestamp,base_yaw.pos,...) and row count of the produced file.",
                       source=cmd)


# ---------------------------------------------------------------------------
# 6 — on real hardware
# ---------------------------------------------------------------------------
def shot_sim2real_log() -> dict:
    """Replay the logged commanded-vs-measured degrees from the physical unit."""
    replay = os.path.join(HERE, "broll_sim2real_replay.py")
    shot = TermShot(title="animacy — Reachy Mini read-back (physical unit, 192.168.1.60)")
    shot.blocks = [
        pace(run_capture(f'"{PY}" "{replay}" data/sim2real/reachy_20260826_214727.json',
                         display="python scripts/video/broll_sim2real_replay.py data/sim2real/reachy_20260826_214727.json",
                         pause_after=1.2), cap=1.2),
    ]
    shot.type_cps = 30.0
    return record_term(shot, "s6_sim2real_readback.mp4", section="6",
                       shows="Commanded vs measured degrees for every axis of the physical Reachy Mini, "
                             "replayed from the logged sim-to-real run (data/sim2real/*.json): each axis "
                             "tracked within a few degrees.",
                       source="scripts/video/broll_sim2real_replay.py over data/sim2real/reachy_20260826_214727.json",
                       notes="Replay of the log written by scripts/reachy_sim2real.py against the real robot "
                             "on 2026-08-26; no robot motion is re-issued.")


def shot_daemon_live() -> dict:
    """Poll the physical robot's daemon while the ambient loop is driving it."""
    poll = os.path.join(HERE, "broll_daemon_poll.py")
    cmd = "python scripts/video/broll_daemon_poll.py --url http://192.168.1.60:8000"
    b = run_capture(f'"{PY}" "{poll}" --url http://192.168.1.60:8000 --collect 45 --seconds 4.5 --hz 6',
                    display=cmd, pause_after=1.3, timeout=180)
    if b.exit_code != 0 or "present_" not in b.output:
        log("  the Reachy daemon did not answer — skipping rather than faking it")
        log(f"  output was: {b.output[:250]}")
        return {}
    b.output_delay = 1.1
    b.output_cps = 520.0            # the rows appear at roughly the rate they were sampled
    shot = TermShot(title="animacy — the robot's own read-back, live (192.168.1.60)")
    shot.type_cps = 26.0
    shot.blocks = [b]
    return record_term(shot, "s6_daemon_live_poll.mp4", section="6",
                       shows="A live read of the physical Reachy Mini's daemon while the ambient "
                             "loop drives it: head yaw/pitch/roll, both antennas, body yaw and head "
                             "translation, in degrees, changing sample to sample. Read-only — the "
                             "poll commands nothing.",
                       source=cmd,
                       notes="The ambient loop rests between clips, so the script polls for 45 s and "
                             "prints the busiest 4.5 s of that read-out; every row is a real sample "
                             "at its real timestamp, and the header says so on screen.")


def shot_sim2real_evidence() -> dict:
    path = os.path.join(ROOT, "docs", "evidence", "reachy_sim2real_20260826.md")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    lines = text.replace("\r\n", "\n").split("\n")
    end = next((i for i, l in enumerate(lines) if l.startswith("## Visual confirmation")), len(lines))
    return record_doc("\n".join(lines[:end]), "s6_sim2real_evidence.mp4",
                      path_label="docs/evidence/reachy_sim2real_20260826.md",
                      meta="physical Reachy Mini Wireless · 192.168.1.60", px_per_sec=96,
                      section="6",
                      shows="The sim-to-real evidence page: the commanded-vs-measured table for every "
                            "segment (look left/right/up/down, roll, brows, lean, body yaw) taken off "
                            "the daemon's own present_head_pose.",
                      source="docs/evidence/reachy_sim2real_20260826.md", max_seconds=15.0)


# ---------------------------------------------------------------------------
# 7 — data
# ---------------------------------------------------------------------------
def shot_data_report() -> dict:
    os.makedirs(ARTIFACTS, exist_ok=True)
    shot = TermShot(title="animacy — the corpus, by licence and speaker")
    shot.blocks = [
        pace(run_capture(f'"{PY}" scripts\\data_report.py > out\\data_report.txt',
                         display="python scripts/data_report.py > out/data_report.txt",
                         pause_after=0.6), cap=2.2),
        run_bash("grep -A 9 '^TOTALS' out/data_report.txt", pause_after=0.9),
        run_bash("grep -E '^(LICENSES|SERIES)' out/data_report.txt", pause_after=1.0),
    ]
    shot.type_cps = 24.0
    return record_term(shot, "s7_data_report.mp4", section="7",
                       shows="`python scripts/data_report.py`: kept clips, captured and valid minutes, "
                             "37 distinct speakers with their shares, and the licence breakdown "
                             "(Public Domain / CC-BY-3.0 / CC-BY-4.0 / CC0) of the captured corpus.",
                       source="python scripts/data_report.py ; grep TOTALS / LICENSES")


def shot_harvest() -> dict:
    """Live harvest counter from squaredcube1, where the harvester actually runs."""
    # the harvester lives at C:\harvest\animacy on that box, in its own venv; the remote
    # login shell is PowerShell, and status.py prints non-ASCII channel names
    remote = (r"$env:PYTHONIOENCODING='utf-8'; Set-Location C:\harvest\animacy; "
              r"C:\harvest\venv\Scripts\python.exe scripts\harvest\status.py")
    real = f'ssh -o BatchMode=yes -o ConnectTimeout=25 squaredcube1 "{remote}"'
    b = run_capture(real, display="ssh squaredcube1 'python scripts/harvest/status.py'",
                    pause_after=1.4, timeout=180, max_lines=12)
    if b.exit_code != 0 or "target" not in b.output:
        log(f"  harvest status unavailable (exit {b.exit_code}) — skipping rather than faking it")
        log(f"  output was: {b.output[:300]}")
        return {}
    b = pace(b, cap=3.0)
    shot = TermShot(title="animacy — harvest, running toward 5,000 hours")
    shot.blocks = [b]
    shot.type_cps = 24.0
    return record_term(shot, "s7_harvest_status.mp4", section="7",
                       shows="The harvester's live status on the box it runs on: hours kept and "
                             "face-valid, hours pushed, the queue still to fetch, the kept-hours "
                             "per wall hour and the ETA to the 5,000-hour target, plus the "
                             "per-language breakdown.",
                       source=real)


# ---------------------------------------------------------------------------
# 8 — the judge
# ---------------------------------------------------------------------------
def shot_scores() -> dict:
    path = os.path.join(ROOT, "docs", "evidence", "grading", "20260827_1501_run3.md")
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().replace("\r\n", "\n").split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("### lamp: held-out lines"))
    end = next(i for i, l in enumerate(lines[start + 1:], start + 1) if l.startswith("### "))
    head = ["# Blind motion grading - run 20260827_1501_run3", "",
            "Judge: kimi-code/k3. Blind: it sees only short clips and a Clip N card.",
            "Lines: sealed held-out lines, authored by the grader and never printed.",
            "`vendor` = the robot maker's own hand-authored clips, on the same rubric.", ""]
    body = "\n".join(head + lines[start:end])
    log(f"  score table: evidence lines {start + 1}..{end}")
    return record_doc(body, "s8_score_table.mp4",
                      path_label="docs/evidence/grading/20260827_1501_run3.md",
                      meta="blind judge · sealed held-out lines", no_gutter=True,
                      fit=True, seconds=10.0, px_per_sec=40, section="8",
                      shows="The published score table for the lamp on the sealed held-out lines: "
                            "animacy's three motion sources beside the `vendor` column of hand-authored "
                            "clips, per movement - level on some, below on others.",
                      source="docs/evidence/grading/20260827_1501_run3.md (lamp, held-out lines)",
                      max_seconds=15.0)


SHOTS = {
    "check": shot_check,
    "robotmd": shot_robotmd,
    "mapping": shot_robotmd_mapping,
    "retarget": shot_retarget,
    "sim2real": shot_sim2real_log,
    "daemon": shot_daemon_live,
    "evidence": shot_sim2real_evidence,
    "data": shot_data_report,
    "harvest": shot_harvest,
    "scores": shot_scores,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", nargs="*", default=list(SHOTS), choices=list(SHOTS))
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    failed = []
    for name in a.shots:
        log(f"\n=== {name} ===")
        try:
            SHOTS[name]()
        except Exception as e:  # noqa: BLE001
            log(f"  FAILED {name}: {type(e).__name__}: {e}")
            failed.append(name)
    if failed:
        log(f"\nfailed shots: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
