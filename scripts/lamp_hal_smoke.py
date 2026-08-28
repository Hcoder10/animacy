"""Prove animacy's Lamp export against Autonomous OS's OWN HAL server.

Starts nothing. Expects a running HAL (their laptop simulator or a real Lamp) at
--url; ``scripts/lamp_hal_sim_start.sh`` boots the simulator (``HAL_SIMULATE=1
HAL_BOARD=sim``, mock motion driver) on Linux/WSL from an unmodified clone.

What it does, in order — every HTTP response is captured verbatim in the report:

1. ``GET /device``, ``/health``, ``/servo`` (their recording list), ``/servo/position``,
   and ``/emotion/status`` if that route is mounted.
2. Builds recordings with animacy: a motion-rich window of each ``--clips`` clip
   (``HumanClip`` -> ``retarget_clip(lamp)`` -> ``to_autonomous_os_csv``) and, with
   ``--say``, the five grader utterances through ``animacy.serve.say(dry_run=True)``.
   Each CSV is checked by ``animacy.export.validate_autonomous_os_csv`` and POSTed to
   ``/servo/upload`` (multipart ``file`` + form ``recording_name``).
3. ``POST /servo/play`` for each, polling ``GET /servo/position`` + ``GET /servo`` while
   it plays; the reported joints are compared with the CSV frame that should be current.
4. Uploads deliberately illegal files (column typo, bad suffix, no timestamp,
   non-numeric, too many rows, empty, over-speed) and records HAL's answer next to
   animacy's validator output — the "mirror" claim.
5. Writes ``<out>/report.json`` (raw) and ``<out>/report.md`` (tables).

``--replay`` re-plays the recordings of a previous run without uploading — after a
HAL restart they are loaded from ``hal/recordings/<name>.csv`` through the vendor's
``resample_recording`` (30 Hz grid + over-speed stretch), which the mock only applies
on that disk path. The expected frames for that mode are computed with the vendor's
own ``recording_timing`` module imported from ``--aos-repo``.

    python scripts/lamp_hal_smoke.py --url http://127.0.0.1:5001 --out out/lamp_hal
    python scripts/lamp_hal_smoke.py --replay --out out/lamp_hal     # after restarting HAL
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from animacy.export import (  # noqa: E402
    AUTONOMOUS_MAX_ROWS,
    to_autonomous_os_csv,
    validate_autonomous_os_csv,
)
from animacy.profile import find_robot  # noqa: E402
from animacy.retarget import retarget_clip  # noqa: E402
from animacy.schema import HumanClip  # noqa: E402

DEFAULT_CLIPS = [
    "interview_with_annie_laurie_gaylor",
    "president_reagan_s_radio_address_on_federal_budg",
    "082810_weeklyaddress_obama",
    "anarkali_honaryar_ghani_has_pledged_to_ensure_wo",
]
PLAYBACK_FPS = 30.0  # hal/drivers/motors/mock_service.py PLAYBACK_FPS == HAL_SERVO_FPS default


# ----------------------------------------------------------------------------- HTTP

class Hal:
    def __init__(self, url: str, timeout: float = 10.0):
        self.base = url.rstrip("/")
        self.s = requests.Session()
        self.timeout = timeout
        self.log: List[dict] = []

    def _rec(self, method: str, path: str, r: requests.Response, extra: Optional[dict] = None) -> dict:
        try:
            body = r.json()
        except ValueError:
            body = r.text
        entry = {"method": method, "path": path, "status": r.status_code, "body": body,
                 "content_type": r.headers.get("content-type")}
        if extra:
            entry.update(extra)
        self.log.append(entry)
        return entry

    def get(self, path: str) -> dict:
        r = self.s.get(self.base + path, timeout=self.timeout)
        return self._rec("GET", path, r)

    def post_json(self, path: str, body: dict) -> dict:
        r = self.s.post(self.base + path, json=body, timeout=self.timeout)
        return self._rec("POST", path, r, {"request": body})

    def upload(self, name: str, text: str, filename: Optional[str] = None) -> dict:
        r = self.s.post(self.base + "/servo/upload",
                        files={"file": (filename or f"{name}.csv", text.encode("utf-8"), "text/csv")},
                        data={"recording_name": name}, timeout=self.timeout)
        return self._rec("POST", "/servo/upload", r, {"request": {"recording_name": name, "filename": filename or f"{name}.csv",
                                                                   "bytes": len(text.encode("utf-8"))}})


# ----------------------------------------------------------------------------- CSV helpers

def parse_csv(text: str) -> Tuple[List[float], List[Dict[str, float]]]:
    times, frames = [], []
    for row in csv.DictReader(io.StringIO(text)):
        times.append(float(row["timestamp"]))
        frames.append({k: float(v) for k, v in row.items() if k != "timestamp"})
    return times, frames


def vendor_resampler(aos_repo: str):
    """Their hal/drivers/motors/recording_timing.resample_recording, or None."""
    if not os.path.isdir(os.path.join(aos_repo, "hal")):
        return None
    sys.path.insert(0, os.path.abspath(aos_repo))
    try:
        from hal.drivers.motors.recording_timing import SERVO_MAX_DPS, resample_recording  # type: ignore
    except Exception as e:  # noqa: BLE001
        print("[warn] vendor recording_timing not importable:", e)
        return None
    return resample_recording, SERVO_MAX_DPS


# ----------------------------------------------------------------------------- building recordings

def motion_window(clip: HumanClip, seconds: float) -> HumanClip:
    """The ``seconds``-long window with the most head motion among fully face-valid windows."""
    f = clip.frames
    n = int(round(seconds * clip.rate_hz))
    if len(f) <= n:
        return clip
    step = int(clip.rate_hz)
    best, best_score = 0, -1.0
    yaw, pitch, fv = f["head_yaw"].to_numpy(), f["head_pitch"].to_numpy(), f["face_valid"].to_numpy()
    for s in range(0, len(f) - n, step):
        if fv[s:s + n].mean() < 0.98:
            continue
        score = float(np.nanstd(yaw[s:s + n]) + np.nanstd(pitch[s:s + n]))
        if score > best_score:
            best, best_score = s, score
    w = f.iloc[best:best + n].reset_index(drop=True).copy()
    w["t"] = (w["t"] - w["t"].iloc[0]).astype(np.float32)
    out = HumanClip.from_frames(w, **{k: v for k, v in clip.meta.items()})
    out.meta["window_start_s"] = float(f["t"].iloc[best])
    return out


def build_clip_recordings(prof, clip_names: List[str], seconds: float, clips_dir: str) -> List[dict]:
    recs = []
    for name in clip_names:
        d = os.path.join(clips_dir, name)
        clip = HumanClip.load(d, audio=False)
        probs = clip.validate()
        if probs:
            print(f"[skip] {name}: {probs}")
            continue
        win = motion_window(clip, seconds)
        table = retarget_clip(win, prof)
        text = to_autonomous_os_csv(table, prof)
        recs.append({"name": "animacy_" + name[:24].rstrip("_"), "source": f"clip window {name} @ {win.meta.get('window_start_s', 0):.1f}s",
                     "kind": "clip", "text": text, "frames": len(table), "authored_s": (len(table) - 1) / PLAYBACK_FPS})
    return recs


def build_say_recordings(prof, source: str, checkpoint: str, seed: int) -> List[dict]:
    from animacy.grade.movements import MOVEMENTS
    from animacy.serve import say

    recs = []
    for mv in MOVEMENTS:
        table = say(mv.text, prof, source=source, dry_run=True, seed=seed, checkpoint=checkpoint,
                    intent=mv.intent_tag, play_audio=False)
        text = to_autonomous_os_csv(table, prof)
        recs.append({"name": f"animacy_say_{mv.key}", "source": f"say({mv.text!r}, source={source}, intent={mv.intent_tag})",
                     "kind": "say", "text": text, "frames": len(table), "authored_s": (len(table) - 1) / PLAYBACK_FPS})
    return recs


def illegal_files(legal_text: str, prof) -> List[dict]:
    """Deliberately broken variants of a legal file, each breaking one rule."""
    lines = legal_text.split("\n")
    header = lines[0]
    joints = [j.name for j in prof.joints]
    out = []

    def variant(key, text, why):
        out.append({"key": key, "why": why, "text": text})

    variant("typo", "\n".join([header.replace("wrist_roll.pos", "wrist_rol.pos")] + lines[1:]),
            "column typo: wrist_rol.pos (unknown joint)")
    variant("bad_suffix", "\n".join([header.replace("base_yaw.pos", "base_yaw")] + lines[1:]),
            "joint column without the .pos suffix")
    variant("no_timestamp", "\n".join([header.replace("timestamp", "time")] + lines[1:]),
            "no timestamp column")
    bad_row = lines[3].split(",")
    bad_row[1] = "abc"
    variant("non_numeric", "\n".join(lines[:3] + [",".join(bad_row)] + lines[4:]), "a non-numeric joint value (row 4)")
    variant("empty", "", "empty file")
    # 20001 data rows of the rest pose (well under the 2 MB cap, so only the row cap fires)
    rest = ",".join(f"{j.rest:.6f}" for j in prof.joints)
    variant("too_many_rows", header + "\n" + "\n".join(f"{i / PLAYBACK_FPS:.6f},{rest}" for i in range(AUTONOMOUS_MAX_ROWS + 1)) + "\n",
            f"{AUTONOMOUS_MAX_ROWS + 1} rows (cap {AUTONOMOUS_MAX_ROWS})")
    # over-speed: +150 units on wrist_roll for three frames then back, at 30 Hz = 4500 units/s
    rows = [ln.split(",") for ln in lines[1:] if ln]
    wr = joints.index("wrist_roll") + 1
    for i in range(10, 13):
        rows[i][wr] = f"{float(rows[i][wr]) + 150.0:.6f}"
    variant("overspeed", header + "\n" + "\n".join(",".join(r) for r in rows) + "\n",
            "legal shape, but a 150-unit step in 1/30 s (4500 units/s) on wrist_roll")
    return out


# ----------------------------------------------------------------------------- play + read back

def play_and_poll(hal: Hal, name: str, expected: List[Dict[str, float]], poll_hz: float,
                  max_seconds: Optional[float] = None, stop_after: Optional[float] = None) -> dict:
    """POST /servo/play then sample /servo/position + /servo until the recording ends."""
    n = len(expected)
    expected_s = (n - 1) / PLAYBACK_FPS
    t_send = time.perf_counter()
    play = hal.post_json("/servo/play", {"recording": name})
    t0 = time.perf_counter()
    samples = []
    seen_playing = False
    ended_at = None
    stop_resp = None
    limit = (max_seconds if max_seconds is not None else expected_s + 4.0)
    while True:
        t_req = time.perf_counter() - t0
        pos = hal.s.get(hal.base + "/servo/position", timeout=hal.timeout).json()["positions"]
        t_resp = time.perf_counter() - t0
        cur = hal.s.get(hal.base + "/servo", timeout=hal.timeout).json()["current"]
        t_mid = 0.5 * (t_req + t_resp)
        samples.append({"t": round(t_mid, 4), "latency_ms": round((t_resp - t_req) * 1000, 1), "current": cur, "positions": pos})
        if cur == name:
            seen_playing = True
        elif seen_playing and ended_at is None:
            ended_at = t_mid
        if stop_after is not None and t_mid >= stop_after and stop_resp is None:
            stop_resp = hal.post_json("/servo/stop", {})
            t_stop = time.perf_counter() - t0
            stop_resp["t"] = round(t_stop, 3)
        if ended_at is not None and t_mid > ended_at + 0.5:
            break
        if stop_resp is not None and t_mid > stop_resp["t"] + 0.6:
            break
        if t_mid > limit:
            break
        time.sleep(max(0.0, 1.0 / poll_hz - (time.perf_counter() - t0 - t_resp)))

    # compare while it reported itself as playing
    joints = list(expected[0].keys())
    exp_arr = np.array([[f[j] for j in joints] for f in expected])
    strict = {j: [] for j in joints}
    on_traj = {j: [] for j in joints}
    lags = []
    k_prev = 0
    for s in samples:
        if s["current"] != name:
            continue
        rep = np.array([s["positions"][j] for j in joints])
        k = max(0, min(n - 1, int(math.floor(s["t"] * PLAYBACK_FPS))))
        for i, j in enumerate(joints):
            strict[j].append(abs(rep[i] - exp_arr[k, i]))
        # Trajectory match: the playback loop's own pacing drifts under host load
        # (its wait-per-frame accumulates overhead), so also find the nearest
        # commanded frame in a monotonic window and measure lag + residual.
        lo, hi = max(0, k_prev - 5), min(n, k_prev + int(3 * PLAYBACK_FPS))
        d = np.abs(exp_arr[lo:hi] - rep)
        km = lo + int(np.argmin(d.max(axis=1)))
        k_prev = km
        s["k_match"] = km
        s["lag_s"] = round(s["t"] - km / PLAYBACK_FPS, 3)
        s["resid_max"] = round(float(np.abs(exp_arr[km] - rep).max()), 3)
        lags.append(s["lag_s"])
        for i, j in enumerate(joints):
            on_traj[j].append(abs(rep[i] - exp_arr[km, i]))
    summary = {}
    for j in joints:
        if strict[j]:
            summary[j] = {"max_err_wallclock": round(max(strict[j]), 3),
                          "max_err_on_trajectory": round(max(on_traj[j]), 3),
                          "mean_err_on_trajectory": round(float(np.mean(on_traj[j])), 3), "n": len(strict[j])}
    return {"play_response": play, "play_roundtrip_ms": round((t0 - t_send) * 1000, 1), "expected_frames": n,
            "expected_duration_s": round(expected_s, 3), "measured_end_s": None if ended_at is None else round(ended_at, 3),
            "seen_playing": seen_playing, "samples": samples, "tracking": summary, "stop_response": stop_resp,
            "lag_first_s": lags[0] if lags else None, "lag_last_s": lags[-1] if lags else None,
            "polls": len(samples)}


# ----------------------------------------------------------------------------- report

def md_readback(rec: dict, every_s: float = 0.5) -> str:
    pb = rec["playback"]
    if not pb["samples"]:
        return "(no samples)\n"
    joints = list(pb["samples"][0]["positions"].keys())
    exp = rec["expected"]
    n = len(exp)
    lines = ["| t (s) | current | lag (s) | " + " | ".join(f"{j[:-4]} cmd / rep" for j in joints) + " |",
             "|---|---|---|" + "---|" * len(joints)]
    next_t = 0.0
    for s in pb["samples"]:
        if s["t"] + 1e-9 < next_t:
            continue
        next_t = s["t"] + every_s
        k = s.get("k_match", max(0, min(n - 1, int(math.floor(s["t"] * PLAYBACK_FPS)))))
        cells = []
        for j in joints:
            rep = s["positions"].get(j)
            cmd = exp[k][j] if s["current"] == rec["name"] else None
            cells.append(f"{'-' if cmd is None else f'{cmd:.1f}'} / {rep:.1f}")
        lines.append(f"| {s['t']:.2f} | {s['current']} | {s.get('lag_s', '-')} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_report(out_dir: str, report: dict, md_name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, md_name.replace(".md", ".json")), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    L = [f"# animacy -> Autonomous OS HAL smoke ({report['mode']}) — {report['started']}", ""]
    L.append(f"HAL: `{report['url']}`  device: `{json.dumps(report['device'].get('body'))}`")
    L.append("")
    if "recordings_before" in report:
        L.append(f"`GET /servo` before upload: {len(report['recordings_before'])} recordings: `{', '.join(report['recordings_before'])}`")
        L.append("")
    for rec in report["recordings"]:
        pb = rec["playback"]
        L.append(f"## {rec['name']}  ({rec['source']})")
        L.append("")
        L.append(f"- frames {rec['frames']}, authored {rec['authored_s']:.2f} s at 30 Hz; animacy validator: `{rec.get('validator', [])}`")
        if "upload" in rec:
            u = rec["upload"]
            L.append(f"- `POST /servo/upload` ({u['request']['bytes']} bytes, recording_name={u['request']['recording_name']}) -> HTTP {u['status']} `{json.dumps(u['body'])}`")
        if rec.get("listed_after_upload") is not None:
            L.append(f"- listed by `GET /servo` after upload: {rec['listed_after_upload']}")
        L.append(f"- `POST /servo/play` -> HTTP {pb['play_response']['status']} `{json.dumps(pb['play_response']['body'])}` (round trip {pb['play_roundtrip_ms']} ms)")
        L.append(f"- expected {pb['expected_frames']} frames = {pb['expected_duration_s']} s; HAL reported `current={rec['name']}` then handed back to idle at "
                 f"{'never (not seen)' if pb['measured_end_s'] is None else str(pb['measured_end_s']) + ' s'}; {pb['polls']} polls")
        if pb.get("stop_response"):
            L.append(f"- `POST /servo/stop` at {pb['stop_response']['t']} s -> HTTP {pb['stop_response']['status']} `{json.dumps(pb['stop_response']['body'])}`")
        if pb["tracking"]:
            L.append(f"- playback pacing: lag vs wall clock {pb.get('lag_first_s')} s at the first sample -> {pb.get('lag_last_s')} s at the last "
                     "(the mock's wait-per-frame loop accumulates host-scheduling overhead; positions are compared on the commanded trajectory)")
            L.append("- reported vs commanded (units = vendor joint units): " + "; ".join(
                f"{j[:-4]} max {v['max_err_on_trajectory']} / mean {v['mean_err_on_trajectory']} (wall-clock-indexed max {v['max_err_wallclock']})"
                for j, v in pb["tracking"].items()))
        L.append("")
        L.append(md_readback(rec))
        L.append("")
    if report.get("rejections"):
        L.append("## Illegal files: HAL's validator vs animacy's")
        L.append("")
        L.append("| file | what is wrong | animacy `validate_autonomous_os_csv` | HAL `POST /servo/upload` |")
        L.append("|---|---|---|---|")
        for r in report["rejections"]:
            L.append(f"| {r['key']} | {r['why']} | `{r['animacy']}` | HTTP {r['hal']['status']} `{json.dumps(r['hal']['body'])}` |")
        L.append("")
    if report.get("emotion"):
        L.append("## /emotion")
        L.append("")
        for e in report["emotion"]:
            L.append(f"- `{e['method']} {e['path']}` -> HTTP {e['status']} `{json.dumps(e['body'])}`")
        L.append("")
    path = os.path.join(out_dir, md_name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    return path


# ----------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:5001")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "lamp_hal"))
    ap.add_argument("--robot", default="lamp")
    ap.add_argument("--clips", default=",".join(DEFAULT_CLIPS))
    ap.add_argument("--clips-dir", default=os.path.join(ROOT, "data", "clips"))
    ap.add_argument("--seconds", type=float, default=10.0, help="clip window length")
    ap.add_argument("--say", action="store_true", help="also the five grader utterances via animacy.serve.say")
    ap.add_argument("--source", default="retrieval")
    ap.add_argument("--checkpoint", default=os.path.join(ROOT, "checkpoints", "v2a"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--poll-hz", type=float, default=20.0)
    ap.add_argument("--extra-csv", action="append", default=[], metavar="NAME=PATH",
                    help="upload an existing CSV (e.g. from `animacy retarget`) and play its first --extra-seconds")
    ap.add_argument("--extra-seconds", type=float, default=8.0)
    ap.add_argument("--skip-illegal", action="store_true")
    ap.add_argument("--replay", action="store_true", help="re-play the recordings of <out>/report.json without uploading")
    ap.add_argument("--aos-repo", default=os.path.join(ROOT, "third_party", "autonomous-os"))
    a = ap.parse_args()

    prof = find_robot(a.robot)
    hal = Hal(a.url)
    try:
        dev = hal.get("/device")
    except requests.RequestException as e:
        print(f"HAL not reachable at {a.url}: {e}")
        return 2
    report = {"mode": "replay" if a.replay else "upload+play", "url": a.url,
              "started": time.strftime("%Y-%m-%d %H:%M:%S"), "device": dev, "health": hal.get("/health"),
              "recordings": [], "recordings_before": hal.get("/servo")["body"].get("available_recordings", []),
              "position_before": hal.get("/servo/position")["body"]}
    print("device:", json.dumps(dev["body"]))
    vend = vendor_resampler(a.aos_repo)

    if a.replay:
        prev_path = os.path.join(a.out, "report.json")
        with open(prev_path, encoding="utf-8") as fh:
            prev = json.load(fh)
        listed = set(report["recordings_before"])
        for r in prev["recordings"]:
            with open(os.path.join(a.out, r["file"]), encoding="utf-8") as fh:
                text = fh.read()
            times, frames = parse_csv(text)
            if vend is None:
                expected, how = frames, "raw CSV rows (vendor resampler unavailable)"
            else:
                expected = vend[0](times, frames, r["name"], PLAYBACK_FPS)
                how = f"vendor resample_recording (SERVO_MAX_DPS={vend[1]})"
            rec = {"name": r["name"], "source": r["source"] + " [from hal/recordings on disk]", "kind": r["kind"], "file": r["file"],
                   "frames": len(frames), "authored_s": (times[-1] - times[0]), "expected_how": how, "expected_frames_n": len(expected),
                   "listed_after_upload": r["name"] in listed, "expected": expected}
            print(f"[replay] {r['name']}: {len(frames)} authored frames -> {len(expected)} expected ({how})")
            cut = r["kind"] == "extra" and rec["authored_s"] > 30
            rec["playback"] = play_and_poll(hal, r["name"], expected, a.poll_hz, max_seconds=a.extra_seconds if cut else None,
                                            stop_after=a.extra_seconds if cut else None)
            report["recordings"].append(rec)
        # also replay the over-speed file if it was uploaded: the disk path stretches it
        for r in prev.get("rejections", []):
            if r["key"] == "overspeed" and r["hal"]["status"] == 200:
                with open(os.path.join(a.out, r["file"]), encoding="utf-8") as fh:
                    text = fh.read()
                times, frames = parse_csv(text)
                expected = vend[0](times, frames, r["name"], PLAYBACK_FPS) if vend else frames
                rec = {"name": r["name"], "source": "over-speed file [from hal/recordings on disk]", "kind": "overspeed", "file": r["file"],
                       "frames": len(frames), "authored_s": times[-1] - times[0], "expected_frames_n": len(expected),
                       "expected_how": "vendor resample_recording" if vend else "raw", "expected": expected, "listed_after_upload": r["name"] in listed}
                print(f"[replay] {r['name']}: {len(frames)} authored frames -> {len(expected)} expected after vendor stretch")
                rec["playback"] = play_and_poll(hal, r["name"], expected, a.poll_hz)
                report["recordings"].append(rec)
        path = write_report(a.out, report, "report_replay.md")
        print("wrote", path)
        return 0

    os.makedirs(a.out, exist_ok=True)
    recs = build_clip_recordings(prof, [c for c in a.clips.split(",") if c], a.seconds, a.clips_dir)
    if a.say:
        recs += build_say_recordings(prof, a.source, a.checkpoint, a.seed)
    for spec in a.extra_csv:
        name, path = spec.split("=", 1)
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        times, frames = parse_csv(text)
        recs.append({"name": name, "source": f"existing file {os.path.relpath(path, ROOT)}", "kind": "extra", "text": text,
                     "frames": len(frames), "authored_s": times[-1] - times[0]})

    max_speed = {j.name: j.max_speed for j in prof.joints}
    for rec in recs:
        rec["file"] = rec["name"] + ".csv"
        with open(os.path.join(a.out, rec["file"]), "w", encoding="utf-8", newline="") as fh:
            fh.write(rec["text"])
        rec["validator"] = validate_autonomous_os_csv(rec["text"], prof.joint_names, max_speed)
        rec["upload"] = hal.upload(rec["name"], rec["text"])
        rec["listed_after_upload"] = rec["name"] in hal.get("/servo")["body"].get("available_recordings", [])
        print(f"[upload] {rec['name']}: {rec['frames']} frames, validator={rec['validator']}, HTTP {rec['upload']['status']} {rec['upload']['body']}, listed={rec['listed_after_upload']}")

    for rec in recs:
        if rec["upload"]["status"] != 200:
            rec["playback"] = {"play_response": {"status": None, "body": "not uploaded"}, "samples": [], "tracking": {}, "polls": 0,
                               "expected_frames": rec["frames"], "expected_duration_s": rec["authored_s"], "measured_end_s": None, "play_roundtrip_ms": 0}
            rec["expected"] = []
            continue
        _, frames = parse_csv(rec["text"])
        rec["expected"] = frames  # same-session upload: the mock plays the uploaded rows one per 1/30 s tick
        extra = rec["kind"] == "extra" and rec["authored_s"] > 30  # only cut short the multi-minute files
        rec["playback"] = play_and_poll(hal, rec["name"], frames, a.poll_hz,
                                        max_seconds=a.extra_seconds + 1.0 if extra else None,
                                        stop_after=a.extra_seconds if extra else None)
        pb = rec["playback"]
        print(f"[play] {rec['name']}: expected {pb['expected_duration_s']} s, ended {pb['measured_end_s']} s, lag {pb.get('lag_first_s')}->{pb.get('lag_last_s')} s, "
              + "; ".join(f"{j[:-4]} max {v['max_err_on_trajectory']}" for j, v in pb["tracking"].items()))
        del rec["text"]
    report["recordings"] = recs

    if not a.skip_illegal and recs:
        legal = open(os.path.join(a.out, recs[0]["file"]), encoding="utf-8").read()
        report["rejections"] = []
        for v in illegal_files(legal, prof):
            name = "animacy_illegal_" + v["key"]
            fn = name + ".csv"
            with open(os.path.join(a.out, fn), "w", encoding="utf-8", newline="") as fh:
                fh.write(v["text"])
            ours = validate_autonomous_os_csv(v["text"], prof.joint_names, max_speed)
            theirs = hal.upload(name, v["text"])
            report["rejections"].append({"key": v["key"], "why": v["why"], "file": fn, "name": name, "animacy": ours, "hal": theirs})
            print(f"[illegal] {v['key']}: animacy={ours} | HAL {theirs['status']} {theirs['body']}")

    emo = []
    st = hal.get("/emotion/status")
    emo.append(st)
    if st["status"] == 200:
        emo.append(hal.post_json("/emotion", {"emotion": "curious"}))
        time.sleep(0.5)
        emo.append(hal.get("/servo"))
    report["emotion"] = emo
    report["recordings_after"] = hal.get("/servo")["body"].get("available_recordings", [])
    report["http_log"] = hal.log
    path = write_report(a.out, report, "report.md")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
