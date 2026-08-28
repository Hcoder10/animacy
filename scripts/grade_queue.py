"""Queue the next blind grading run for when its inputs land (then run it and compare with the baseline).

    python scripts/grade_queue.py --baseline data/grading/20260826_2320 --since 37eca54 \\
        --checkpoint checkpoints/v2a --label "run 2: fitted mapping + v2a" [--poll 60] [--max-hours 12] [--dry-run]

Fires when BOTH hold, and have held unchanged for --settle seconds:
  1. a commit after --since touches robots/<robot>/ROBOT.md for any graded robot (the fitted mapping landed), and
  2. <checkpoint>/REPORT.md exists, <checkpoint> has a2m.pt or a2m_ar.pt, and web/models/model.json differs
     from what it was when this queue started (the new bundle was exported).
Then: scripts/grade_run.py with the same movements/rubric/seeds as the baseline (model 2 seeds, deterministic
sources 1) --compare <baseline>, into data/grading/<timestamp>_run2. State + log: data/grading/queue_state.json,
data/grading/queue.log. Run detached (Start-Process) so it outlives the shell that started it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ROBOTS = ["lamp", "reachy_mini"]


def sha1(path: str) -> str:
    if not os.path.isfile(path):
        return ""
    return hashlib.sha1(open(path, "rb").read()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()


def conditions(since: str, checkpoint: str, model_json_ref: str, robots) -> dict:
    ck = os.path.join(ROOT, checkpoint)
    robot_md = [f"robots/{r}/ROBOT.md" for r in robots]
    commits = git("log", "--format=%h %ci %s", f"{since}..HEAD", "--", *robot_md)
    model_json = os.path.join(ROOT, "web", "models", "model.json")
    weights = [w for w in ("a2m.pt", "a2m_ar.pt") if os.path.isfile(os.path.join(ck, w))]
    c = {
        "mapping_commit": commits.splitlines()[0] if commits else None,
        "report_md": os.path.isfile(os.path.join(ck, "REPORT.md")),
        "weights": weights,
        "model_json_changed": sha1(model_json) != model_json_ref,
        "model_json_sha1": sha1(model_json)[:12],
    }
    c["mapping_ok"] = bool(c["mapping_commit"])
    c["bundle_ok"] = bool(c["report_md"] and weights and c["model_json_changed"])
    c["ready"] = c["mapping_ok"] and c["bundle_ok"]
    c["fingerprint"] = json.dumps([c["mapping_commit"], sha1(os.path.join(ck, "REPORT.md"))[:12],
                                   [sha1(os.path.join(ck, w))[:12] for w in weights], c["model_json_sha1"],
                                   [sha1(os.path.join(ROOT, p))[:12] for p in robot_md]])
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="baseline run dir (data/grading/<run>)")
    ap.add_argument("--since", required=True, help="commit; a ROBOT.md commit after it counts as 'the mapping landed'")
    ap.add_argument("--checkpoint", default="checkpoints/v2a")
    ap.add_argument("--label", default="run 2: fitted mapping + v2a bundle")
    ap.add_argument("--robots", nargs="+", default=ROBOTS)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--settle", type=int, default=180, help="both conditions must hold unchanged this long")
    ap.add_argument("--max-hours", type=float, default=12.0)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true", help="print the condition status once and exit")
    a = ap.parse_args()

    log_path = os.path.join(ROOT, "data", "grading", "queue.log")
    state_path = os.path.join(ROOT, "data", "grading", "queue_state.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(msg: str) -> None:
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    model_json_ref = sha1(os.path.join(ROOT, "web", "models", "model.json"))
    log(f"[queue] armed: since={a.since} checkpoint={a.checkpoint} baseline={a.baseline} model.json={model_json_ref[:12]} "
        f"poll={a.poll}s settle={a.settle}s max={a.max_hours}h")
    t_end = time.time() + a.max_hours * 3600
    stable_since = None
    stable_fp = None
    while True:
        c = conditions(a.since, a.checkpoint, model_json_ref, a.robots)
        with open(state_path, "w", encoding="utf-8") as fh:
            json.dump({"checked": dt.datetime.now().isoformat(timespec="seconds"), "conditions": c, "args": vars(a)}, fh, indent=1)
        if a.dry_run:
            print(json.dumps(c, indent=1))
            return 0
        if c["ready"]:
            if stable_fp != c["fingerprint"]:
                stable_fp, stable_since = c["fingerprint"], time.time()
                log(f"[queue] both conditions hold: mapping commit '{c['mapping_commit']}', bundle {c['weights']} "
                    f"model.json {c['model_json_sha1']}; settling {a.settle}s")
            elif time.time() - stable_since >= a.settle:
                break
        else:
            stable_fp, stable_since = None, None
        if time.time() > t_end:
            log(f"[queue] gave up after {a.max_hours}h: mapping_ok={c['mapping_ok']} bundle_ok={c['bundle_ok']} "
                f"(report_md={c['report_md']} weights={c['weights']} model_json_changed={c['model_json_changed']})")
            return 2
        time.sleep(a.poll)

    out = os.path.join(ROOT, "data", "grading", dt.datetime.now().strftime("%Y%m%d_%H%M") + "_run2")
    label = f"{a.label} (mapping {c['mapping_commit'].split()[0]}, model.json {c['model_json_sha1']})"
    cmd = [a.python, "-u", os.path.join(HERE, "grade_run.py"), "--robots", *a.robots, "--sources", "model", "retrieval",
           "envelope", "--seeds", "2", "--seeds-deterministic", "1", "--checkpoint", a.checkpoint, "--compare", a.baseline,
           "--label", label, "--out", out]
    log("[queue] launching: " + " ".join(cmd))
    with open(out + ".log", "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, check=False).returncode
    log(f"[queue] run finished with exit {rc}: {out} (log {out}.log)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
