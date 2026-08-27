"""Supervisor: runs crawl / fetch / workers / push as child processes, restarts any that exit.

    python scripts/harvest/daemon.py --n 16 [--repo squaredcuber/animacy-human-motion-large] [--no-crawl]

Logs: <ROOT>/logs/{crawl,fetch,workers,push}.log. Stop everything: kill this process (children are
killed on exit) or create <ROOT>/STOP.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import common as C

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))


def spawn(name: str, args: list) -> subprocess.Popen:
    log = open(os.path.join(C.LOGS, f"{name}.log"), "a", encoding="utf-8")
    log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} start {name}\n")
    log.flush()
    return subprocess.Popen([PY, os.path.join(HERE, f"{name}.py"), *args], stdout=log, stderr=subprocess.STDOUT,
                            cwd=C.REPO, env=C.child_env())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--repo", default=os.environ.get("HARVEST_HF_REPO", "squaredcuber/animacy-human-motion-large"))
    p.add_argument("--buffer", type=int, default=0, help="fetch buffer (default 2*n + 4)")
    p.add_argument("--no-crawl", action="store_true")
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--min-push-hours", type=float, default=50.0)
    a = p.parse_args(argv)
    C.ensure_dirs()
    stop_file = os.path.join(C.ROOT, "STOP")
    if os.path.exists(stop_file):
        os.remove(stop_file)
    jobs = {
        "fetch": ["--loop", "--buffer", str(a.buffer or 2 * a.n + 4)],
        "workers": ["--n", str(a.n)],
    }
    if not a.no_crawl:
        jobs["crawl"] = ["--loop"]
    if not a.no_push:
        jobs["push"] = ["--loop", "--repo", a.repo, "--min-hours", str(a.min_push_hours)]
    procs = {k: spawn(k, v) for k, v in jobs.items()}
    C.log("daemon: started " + ", ".join(f"{k} pid {p.pid}" for k, p in procs.items()))
    try:
        while True:
            time.sleep(30)
            if os.path.exists(stop_file):
                C.log("daemon: STOP file found")
                break
            for k, pr in list(procs.items()):
                if pr.poll() is not None:
                    C.log(f"daemon: {k} exited rc={pr.returncode}; restarting in 60 s")
                    time.sleep(60)
                    procs[k] = spawn(k, jobs[k])
    finally:
        for k, pr in procs.items():
            try:
                pr.terminate()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
