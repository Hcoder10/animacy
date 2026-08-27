#!/usr/bin/env python
"""Score a ROBOT.md mapping (docs/RETARGET.md §evaluation).

    python scripts/retarget_eval.py --robot lamp                 # working-tree mapping
    python scripts/retarget_eval.py --robot lamp --before HEAD   # git version side by side
    python scripts/retarget_eval.py --robot reachy_mini --before HEAD --markdown docs/_eval_reachy.md

Per mapping version:
  (a) envelope match — per-joint |.|p95 of the retargeted human corpus over the
      vendor's native-clip |.|p95 (1.00 = matched); score = exp(−mean |log ratio|)
  (b) gaze error under lean (lamp: URDF FK of the head's look axis)
  (c) legality — speed-cap and limit violations, offline (retarget_clip) and
      live (LiveRetargeter at the robot rate); must be 0
  (d) stillness (fraction of frames < 5 units/s) and velocity-histogram W1
      distance vs the vendor library, per joint
  (e) JS-parity readiness — which v1.1 features the profile uses, whether
      web/js/retarget.js implements them, whether web/robots/<name>.json is
      a fresh export, and (Reachy) whether both brows up gives a symmetric
      outward antenna splay
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from animacy.profile import Profile, find_robot, load_profile  # noqa: E402
from animacy.retarget import LiveRetargeter, retarget_clip  # noqa: E402
from animacy.retarget_fit import (  # noqa: E402
    DEFAULT_EXCLUDE, STILL_UNITS_PER_S, fmt_table, gaze_error_cases, joint_stats, library_center, load_human_clips,
    load_native_clips, load_urdf, velocity_w1,
)
from animacy.schema import HumanClip, empty_frames  # noqa: E402

LEAN_CASES = [
    {"head_x": 50.0}, {"head_x": 100.0}, {"head_x": 150.0},
    {"torso_lean_fwd": 10.0}, {"torso_lean_fwd": 20.0},
    {"head_z": 50.0}, {"head_z": -50.0}, {"mouth_open": 1.0},
    {"head_x": 100.0, "torso_lean_fwd": 15.0, "head_z": 30.0, "mouth_open": 0.5},
]
GAZE = {"lamp": "wrist_pitch"}


def profile_at(robot: str, rev: str) -> Profile:
    """The robot's ROBOT.md at git revision ``rev`` (paths resolve against the working-tree robot dir)."""
    live = find_robot(robot)
    rel = os.path.relpath(live.path, ROOT).replace(os.sep, "/")
    text = subprocess.run(["git", "show", f"{rev}:{rel}"], capture_output=True, text=True, cwd=ROOT, check=True, encoding="utf-8").stdout
    tmp = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8")
    tmp.write(text)
    tmp.close()
    prof = load_profile(tmp.name)
    os.unlink(tmp.name)
    prof.path = live.path
    return prof


def live_table(clip: HumanClip, prof: Profile, mode: str) -> pd.DataFrame:
    rt = LiveRetargeter(prof, mode)
    dt = 1.0 / prof.rate_hz
    rows = [rt.step({c: float(v) for c, v in r.items()}, dt) for r in clip.frames.to_dict("records")]
    out = pd.DataFrame(rows)
    out.insert(0, "t", np.arange(len(out)) / prof.rate_hz)
    return out


def violations(tables, prof: Profile):
    """Speed is judged on the robot's own grid (frames are played 1/rate_hz
    apart by every exporter/sink), so per-frame |Δ| must stay ≤ max_speed/rate."""
    speed = limit = 0
    step = 1.0 / prof.rate_hz
    for tb in tables:
        for j in prof.joints:
            v = tb[j.name].to_numpy()
            speed += int(np.sum(np.abs(np.diff(v)) / step > j.max_speed * (1 + 1e-6)))
            limit += int(np.sum((v < j.min - 1e-6) | (v > j.max + 1e-6)))
    return speed, limit


def evaluate(prof: Profile, mode: str, humans, native, js_text: str):
    joints = prof.joint_names
    rest = {j.name: j.rest for j in prof.joints}
    center = library_center(list(native.values()), joints)
    vendor = joint_stats(list(native.values()), joints, center)
    offline = [retarget_clip(c, prof, mode) for c in humans.values()]
    live = [live_table(c, prof, mode) for c in humans.values()]
    off_st = joint_stats(offline, joints, rest)
    live_st = joint_stats(live, joints, rest)
    res = {"joints": {}, "features": {}}
    logs = []
    for j in joints:
        if j not in vendor or j not in off_st or vendor[j].abs_p95 < 1e-9:
            continue
        ratio = off_st[j].abs_p95 / vendor[j].abs_p95
        logs.append(abs(math.log(max(ratio, 1e-9))))
        res["joints"][j] = {
            "vendor_p95": vendor[j].abs_p95, "p95": off_st[j].abs_p95, "ratio": ratio,
            "vendor_vel_p95": vendor[j].vel_p95, "vel_p95": off_st[j].vel_p95,
            "vendor_still": vendor[j].still, "still": off_st[j].still, "live_still": live_st[j].still,
            "w1": velocity_w1(off_st[j], vendor[j]),
        }
    res["envelope_score"] = math.exp(-float(np.mean(logs))) if logs else float("nan")
    res["speed_viol_offline"], res["limit_viol_offline"] = violations(offline, prof)
    res["speed_viol_live"], res["limit_viol_live"] = violations(live, prof)
    res["still_overall"] = float(np.mean([s.still for s in off_st.values()]))
    res["vendor_still_overall"] = float(np.mean([s.still for s in vendor.values()]))
    if prof.name in GAZE:
        try:
            urdf = load_urdf(prof)
            errs = gaze_error_cases(prof, mode, urdf, LEAN_CASES, GAZE[prof.name])
            res["gaze"] = [(ch, e) for ch, e, _ in errs]
            res["gaze_max_abs"] = max(abs(e) for _, e in res["gaze"])
        except ImportError:
            res["gaze"] = None
    # (e) readiness
    used = {"spring": False, "idle": False, "soft_limit": False}
    for m in prof.mapping(mode).values():
        used["spring"] |= m.spring is not None
        used["idle"] |= m.idle is not None
        used["soft_limit"] |= m.soft_limit is not None
    impl = {"spring": "springStep" in js_text, "idle": "idleValue" in js_text, "soft_limit": "softClip" in js_text}
    res["features"] = {k: {"used": used[k], "js": impl[k]} for k in used}
    web_json = os.path.join(ROOT, "web", "robots", f"{prof.name}.json")
    fresh = json.loads(json.dumps(find_robot(prof.name).to_web_json()))
    res["web_json_fresh"] = os.path.exists(web_json) and json.load(open(web_json, encoding="utf-8")) == fresh
    if "antenna_left" in joints and "antenna_right" in joints:
        f = empty_frames(1)
        f["brow_l"] = 1.0
        f["brow_r"] = 1.0
        from animacy.retarget import raw_joint_targets

        q = raw_joint_targets(f, prof, mode).iloc[0]
        res["antenna_symmetric_outward"] = bool(q["antenna_right"] > 5 and abs(q["antenna_left"] + q["antenna_right"]) < 1e-6)
        res["antenna_values_brows_up"] = (float(q["antenna_left"]), float(q["antenna_right"]))
    return res


def render(name: str, versions):
    out = []
    labels = [v for v, _ in versions]
    out.append(f"### {name}: {' vs '.join(labels)}\n")
    joints = list(versions[-1][1]["joints"])
    hdr = ["joint", "vendor |.|p95"] + [f"{l} |.|p95 (ratio)" for l in labels] + ["vendor vel p95"] + [f"{l} vel p95" for l in labels]
    rows = []
    for j in joints:
        r = [j, f"{versions[-1][1]['joints'][j]['vendor_p95']:.1f}"]
        for _, res in versions:
            d = res["joints"].get(j)
            r.append(f"{d['p95']:.1f} ({d['ratio']:.2f})" if d else "-")
        r.append(f"{versions[-1][1]['joints'][j]['vendor_vel_p95']:.0f}")
        for _, res in versions:
            d = res["joints"].get(j)
            r.append(f"{d['vel_p95']:.0f}" if d else "-")
        rows.append(r)
    out.append("**(a) envelope match** (retargeted human corpus vs vendor native clips)\n")
    out.append(fmt_table(rows, hdr))
    out.append("\nenvelope score exp(−mean|log ratio|): " + ", ".join(f"{l} = {res['envelope_score']:.2f}" for l, res in versions) + "\n")
    if versions[-1][1].get("gaze"):
        out.append("**(b) gaze elevation error under lean** (URDF FK of the head's look axis, deg; 0 = still on the person)\n")
        rows = []
        for i, (ch, _) in enumerate(versions[-1][1]["gaze"]):
            rows.append([", ".join(f"{k}={v:g}" for k, v in ch.items())] + [f"{res['gaze'][i][1]:+.2f}" for _, res in versions])
        out.append(fmt_table(rows, ["channels"] + labels))
        out.append("\nmax |error|: " + ", ".join(f"{l} = {res['gaze_max_abs']:.2f} deg" for l, res in versions) + "\n")
    out.append("**(c) legality** (speed-cap / limit violations over the whole corpus, offline and live)\n")
    out.append(fmt_table([[l, str(res["speed_viol_offline"]), str(res["limit_viol_offline"]), str(res["speed_viol_live"]), str(res["limit_viol_live"])] for l, res in versions],
                         ["version", "speed offline", "limit offline", "speed live", "limit live"]))
    out.append("\n**(d) stillness & velocity histogram** (still = fraction of frames < 5 units/s; W1 = velocity-distribution distance to the vendor, relative to the vendor mean speed)\n")
    hdr = ["joint", "vendor still"] + [f"{l} still" for l in labels] + [f"{l} W1" for l in labels]
    rows = []
    for j in joints:
        r = [j, f"{versions[-1][1]['joints'][j]['vendor_still']:.2f}"]
        for _, res in versions:
            d = res["joints"].get(j)
            r.append(f"{d['still']:.2f}" if d else "-")
        for _, res in versions:
            d = res["joints"].get(j)
            r.append(f"{d['w1']:.2f}" if d else "-")
        rows.append(r)
    out.append(fmt_table(rows, hdr))
    out.append("\noverall stillness: vendor " + f"{versions[-1][1]['vendor_still_overall']:.2f}, " + ", ".join(f"{l} {res['still_overall']:.2f}" for l, res in versions) + "\n")
    out.append("**(e) JS-parity readiness** (working tree)\n")
    res = versions[-1][1]
    rows = [[k, "yes" if v["used"] else "no", "yes" if v["js"] else "NO"] for k, v in res["features"].items()]
    rows.append(["web/robots/<name>.json fresh", "-", "yes" if res["web_json_fresh"] else "NO"])
    if "antenna_symmetric_outward" in res:
        l, r_ = res["antenna_values_brows_up"]
        rows.append([f"both brows up → antennas (L, R) = ({l:.0f}, {r_:.0f})", "-", "symmetric outward" if res["antenna_symmetric_outward"] else "NOT symmetric"])
    out.append(fmt_table(rows, ["feature", "used by profile", "implemented in web/js/retarget.js"]))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True)
    ap.add_argument("--mode", default="default")
    ap.add_argument("--clips", default=os.path.join(ROOT, "data", "clips"))
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--before", help="git revision of ROBOT.md to compare against (e.g. HEAD)")
    ap.add_argument("--markdown", help="write the report here as well")
    a = ap.parse_args()
    after = find_robot(a.robot)
    humans = load_human_clips(a.clips, a.names, DEFAULT_EXCLUDE)
    native = load_native_clips(after)
    js_text = open(os.path.join(ROOT, "web", "js", "retarget.js"), encoding="utf-8").read()
    versions = []
    if a.before:
        versions.append((f"before ({a.before})", evaluate(profile_at(a.robot, a.before), a.mode, humans, native, js_text)))
    versions.append(("after", evaluate(after, a.mode, humans, native, js_text)))
    report = render(after.name, versions)
    print(f"human corpus: {len(humans)} clips, {sum(len(c) for c in humans.values())} frames; vendor: {len(native)} native clips\n")
    print(report)
    if a.markdown:
        with open(a.markdown, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
        print("wrote", a.markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
