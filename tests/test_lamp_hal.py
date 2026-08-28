"""animacy's Lamp export against a running Autonomous OS HAL (simulator or unit).

Skipped unless a HAL answers at ``LAMP_HAL_URL`` (default http://127.0.0.1:5001);
``scripts/lamp_hal_sim_start.sh`` boots their laptop simulator on Linux/WSL.
Uploads land in the server's ``hal/recordings/`` under ``animacy_test_*`` names.
"""
from __future__ import annotations

import csv
import io
import math
import os
import time

import numpy as np
import pandas as pd
import pytest

requests = pytest.importorskip("requests")

from animacy.export import to_autonomous_os_csv, validate_autonomous_os_csv  # noqa: E402
from animacy.profile import find_robot  # noqa: E402

HAL = os.environ.get("LAMP_HAL_URL", "http://127.0.0.1:5001").rstrip("/")
FPS = 30.0

try:
    _r = requests.get(HAL + "/servo", timeout=1.5)
    _r.raise_for_status()
    _r.json()["available_recordings"]
except Exception as _e:  # noqa: BLE001
    pytest.skip(f"no Autonomous OS HAL at {HAL}: {_e}", allow_module_level=True)


@pytest.fixture(scope="module")
def prof():
    return find_robot("lamp")


def _table(prof, seconds: float = 2.0, amp: float = 20.0) -> pd.DataFrame:
    """rest pose + a slow sine on wrist_roll / base_yaw, well under max_speed."""
    n = int(seconds * FPS) + 1
    t = np.arange(n) / FPS
    d = {"t": t}
    for j in prof.joints:
        d[j.name] = np.full(n, j.rest, dtype=float)
    d["wrist_roll"] = d["wrist_roll"] + amp * np.sin(2 * math.pi * 0.5 * t)
    d["base_yaw"] = d["base_yaw"] + 0.5 * amp * np.sin(2 * math.pi * 0.5 * t)
    return pd.DataFrame(d)


def _upload(name: str, text: str):
    return requests.post(HAL + "/servo/upload", files={"file": (name + ".csv", text.encode("utf-8"), "text/csv")},
                         data={"recording_name": name}, timeout=10)


def test_animacy_csv_is_accepted_and_listed(prof):
    text = to_autonomous_os_csv(_table(prof), prof)
    assert validate_autonomous_os_csv(text, prof.joint_names, {j.name: j.max_speed for j in prof.joints}) == []
    r = _upload("animacy_test_accept", text)
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "ok"}
    listed = requests.get(HAL + "/servo", timeout=5).json()["available_recordings"]
    assert "animacy_test_accept" in listed


@pytest.mark.parametrize("key", ["typo", "bad_suffix", "no_timestamp", "non_numeric", "empty"])
def test_rejections_match_animacy_validator(prof, key):
    text = to_autonomous_os_csv(_table(prof, seconds=0.5), prof)
    lines = text.split("\n")
    header = lines[0]
    if key == "typo":
        text = "\n".join([header.replace("wrist_roll.pos", "wrist_rol.pos")] + lines[1:])
    elif key == "bad_suffix":
        text = "\n".join([header.replace("base_yaw.pos", "base_yaw")] + lines[1:])
    elif key == "no_timestamp":
        text = "\n".join([header.replace("timestamp", "time")] + lines[1:])
    elif key == "non_numeric":
        row = lines[3].split(",")
        row[1] = "abc"
        text = "\n".join(lines[:3] + [",".join(row)] + lines[4:])
    elif key == "empty":
        text = ""
    ours = validate_autonomous_os_csv(text, prof.joint_names)
    assert ours, "animacy's validator must reject this file"
    r = _upload("animacy_test_" + key, text)
    assert r.status_code == 400, r.text
    # the first message animacy produces is the vendor's own wording (hal/routes/servo.py)
    assert r.json()["detail"] == ours[0]


def test_playback_tracks_the_uploaded_frames(prof):
    table = _table(prof, seconds=3.0)
    text = to_autonomous_os_csv(table, prof)
    name = "animacy_test_track"
    assert _upload(name, text).status_code == 200
    rows = [{k: float(v) for k, v in row.items() if k != "timestamp"} for row in csv.DictReader(io.StringIO(text))]
    r = requests.post(HAL + "/servo/play", json={"recording": name}, timeout=10)
    t0 = time.perf_counter()
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    errs, playing_samples, ended = [], 0, None
    while time.perf_counter() - t0 < 6.0:
        t = time.perf_counter() - t0
        pos = requests.get(HAL + "/servo/position", timeout=5).json()["positions"]
        cur = requests.get(HAL + "/servo", timeout=5).json()["current"]
        if cur == name:
            playing_samples += 1
            k = max(0, min(len(rows) - 1, int(t * FPS)))
            for j, v in rows[k].items():
                # one frame of polling jitter either side
                errs.append(min(abs(pos[j] - rows[kk][j]) for kk in range(max(0, k - 1), min(len(rows), k + 2))))
        elif playing_samples and ended is None:
            ended = t
            break
        time.sleep(0.05)
    assert playing_samples >= 10, "HAL never reported the recording as current"
    assert max(errs) < 2.0, f"reported joints diverged from the CSV: max err {max(errs):.2f}"
    assert ended is not None and abs(ended - 3.0) < 0.6, f"recording should hand back to idle at ~3.0 s, got {ended}"
