"""Drain-and-restart the running workers.py so it picks up new code WITHOUT killing in-flight captures.

    python scripts/harvest/reload_workers.py [--timeout-min 90]

New workers.py instances honour <ROOT>/RELOAD themselves; this script exists for an instance that
predates that hook: it parks every 'fetched' item as 'fetched_hold' so the old workers find nothing
new, waits until no item is 'capturing', kills the old workers.py process tree (all captures done),
restores the held items, and lets daemon.py respawn workers.py on the current code.
"""
from __future__ import annotations

import argparse
import subprocess
import time

import common as C


def kill_workers() -> int:
    if not C.IS_WIN:
        r = subprocess.run(["pkill", "-f", "harvest/workers.py"], capture_output=True)
        return 0 if r.returncode in (0, 1) else r.returncode
    ps = ("$w = Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*harvest\\workers.py*' }; "
          "foreach ($p in $w) { taskkill /PID $p.ProcessId /T /F | Out-Null; 'killed ' + $p.ProcessId }")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    print(r.stdout.strip())
    return r.returncode


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timeout-min", type=float, default=90.0)
    a = p.parse_args(argv)
    con = C.db()
    n = con.execute("UPDATE items SET state='fetched_hold' WHERE state='fetched'").rowcount
    C.log(f"held {n} fetched items; waiting for in-flight captures to finish")
    t0 = time.time()
    try:
        while True:
            k = con.execute("SELECT COUNT(*) n FROM items WHERE state='capturing'").fetchone()["n"]
            if k == 0:
                break
            if time.time() - t0 > a.timeout_min * 60:
                C.log(f"timeout with {k} still capturing; killing anyway (they go back to fetched and are recaptured)")
                break
            C.log(f"  {k} items still capturing ({(time.time() - t0) / 60:.0f} min)")
            time.sleep(30)
        kill_workers()
    finally:
        con.execute("UPDATE items SET state='fetched' WHERE state='fetched_hold'")
        C.log("released held items; daemon will respawn workers.py within 60 s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
