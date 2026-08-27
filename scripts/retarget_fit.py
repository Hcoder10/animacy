#!/usr/bin/env python
"""Fit a ROBOT.md mapping's gains to the vendor's own motion envelope (data → numbers).

    python scripts/retarget_fit.py --robot lamp            # print before/after tables
    python scripts/retarget_fit.py --robot lamp --write    # also write the gains into ROBOT.md
    python scripts/retarget_fit.py --robot reachy_mini --write

Per joint it measures (a) the vendor native clips' excursion around the
library median (p5/p50/p95, |.|p95, velocity p95) and (b) the same for the
real human clips under the current ``default`` mapping, then proposes one gain
multiplier per joint so (b)'s |.|p95 equals (a)'s, capped by the mapping's
headroom and by ``[--min-mult, --max-mult]``. For the lamp it also
linearises the URDF FK of the head's pointing direction and rewrites the
gaze-compensation terms on ``wrist_pitch`` (``docs/RETARGET.md`` §gaze).

Joints in ``--skip`` keep their gains (function-transfer joints such as the
Reachy antennas are fitted to expression anchors by hand — see docs/RETARGET.md).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from animacy.profile import find_robot, load_profile  # noqa: E402
from animacy.retarget_fit import (  # noqa: E402
    DEFAULT_EXCLUDE, current_gains, fmt_table, gaze_compensation, gaze_error_cases, gaze_jacobian, joint_stats,
    library_center, load_human_clips, load_native_clips, load_urdf, propose_multipliers, retarget_tables,
    rewrite_gains, scaled_gains, unclamped_targets,
)

# Which joints carry the gaze and which move the head *around* on each body.
GAZE = {"lamp": {"gaze_joint": "wrist_pitch", "body_joints": ("base_pitch", "elbow_pitch")}}
# Function-transfer joints whose envelope is not comparable 1:1 with the vendor's:
# Reachy's antennas are fitted to expression anchors (docs/RETARGET.md §antennas).
DEFAULT_SKIP = {"reachy_mini": ("antenna_left", "antenna_right"), "lamp": ()}
LEAN_CASES = [
    {"head_x": 50.0}, {"head_x": 100.0}, {"head_x": 150.0},
    {"torso_lean_fwd": 10.0}, {"torso_lean_fwd": 20.0},
    {"head_z": 50.0}, {"head_z": -50.0}, {"mouth_open": 1.0},
    {"head_x": 100.0, "torso_lean_fwd": 15.0, "head_z": 30.0, "mouth_open": 0.5},
]


def stats_rows(st, joints):
    rows = []
    for j in joints:
        if j not in st:
            continue
        s = st[j]
        rows.append([j, f"{s.p5:.1f}", f"{s.p50:.1f}", f"{s.p95:.1f}", f"{s.abs_p95:.1f}", f"{s.abs_p99:.1f}", f"{s.vel_p95:.0f}", f"{s.vel_p99:.0f}", f"{s.still:.2f}"])
    return rows


HDR = ["joint", "p5", "p50", "p95", "|.|p95", "|.|p99", "vel p95", "vel p99", "still"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True)
    ap.add_argument("--mode", default="default")
    ap.add_argument("--clips", default=os.path.join(ROOT, "data", "clips"))
    ap.add_argument("--names", nargs="*", help="human clip names (default: all minus the known-bad ones)")
    ap.add_argument("--vendor", nargs="*", help="native clip names (default: all)")
    ap.add_argument("--min-mult", type=float, default=0.5)
    ap.add_argument("--max-mult", type=float, default=3.0)
    ap.add_argument("--headroom", type=float, default=0.9)
    ap.add_argument("--vel-cap", type=float, default=1.25, help="scaled velocity p95 <= this x vendor velocity p95 (0 = off)")
    ap.add_argument("--passes", type=int, default=3, help="fixed-point passes re-measuring the actual pipeline")
    ap.add_argument("--skip", nargs="*", help="joints to leave untouched (default per robot)")
    ap.add_argument("--no-gaze", action="store_true", help="do not rewrite the gaze-compensation terms")
    ap.add_argument("--write", action="store_true", help="write the fitted gains into ROBOT.md")
    ap.add_argument("--stamp", default=_dt.date.today().isoformat())
    a = ap.parse_args()

    prof = find_robot(a.robot)
    joints = prof.joint_names
    skip = tuple(a.skip) if a.skip is not None else DEFAULT_SKIP.get(prof.name, ())

    native = load_native_clips(prof, a.vendor)
    center = library_center(list(native.values()), joints)
    vendor = joint_stats(list(native.values()), joints, center)
    print(f"# {prof.name}: vendor envelope from {len(native)} native clips (values − library median)")
    print("library median − rest: " + ", ".join(f"{j}={center[j] - prof.joint(j).rest:+.1f}" for j in joints))
    print(fmt_table(stats_rows(vendor, joints), HDR))

    humans = load_human_clips(a.clips, a.names, DEFAULT_EXCLUDE)
    print(f"\n# human clips: {', '.join(f'{n} ({len(c)} frames, kept {c.meta.get('kept_fraction', 1):.2f})' for n, c in humans.items())}")
    rest = {j.name: j.rest for j in prof.joints}
    # the gaze joint's compensation terms are derived from the body joints (below),
    # so the amplitude/headroom fit of that joint looks at its expressive terms only
    exclude = {}
    if prof.name in GAZE and not a.no_gaze:
        g = GAZE[prof.name]
        body_channels = {t.from_ for b in g["body_joints"] if b in prof.mapping(a.mode) for t in prof.mapping(a.mode)[b].terms()}
        exclude[g["gaze_joint"]] = sorted(body_channels)
    lin = [unclamped_targets(c, prof, a.mode, exclude) for c in humans.values()]
    cur_lin = joint_stats(lin, joints, {})
    before = joint_stats(list(retarget_tables(humans, prof, a.mode).values()), joints, rest)
    print(f"\n# BEFORE: human clips through the current `{a.mode}` mapping (values − rest)")
    print(fmt_table(stats_rows(before, joints), HDR))

    import copy

    urdf = jac = None
    if prof.name in GAZE and not a.no_gaze:
        g = GAZE[prof.name]
        urdf = load_urdf(prof)
        jac = gaze_jacobian(urdf, prof, [g["gaze_joint"], *g["body_joints"]])
        print("\n# gaze Jacobian at rest (deg elevation / deg azimuth per joint unit): " + ", ".join(f"{k}=({v[0]:+.3f}, {v[1]:+.3f})" for k, v in jac.items()))
        errs = gaze_error_cases(prof, a.mode, urdf, LEAN_CASES, g["gaze_joint"])
        print("gaze elevation error under lean, CURRENT mapping: " + "; ".join(f"{ch}: {e:+.1f} deg" for ch, e, _ in errs))

    def with_gains(gains):
        """A copy of the profile with every term gain replaced."""
        p2 = copy.deepcopy(prof)
        for jn, ch in gains.items():
            m = p2.mapping(a.mode)[jn]
            if m.mix is not None:
                for t in m.mix:
                    t.gain = ch[t.from_]
            else:
                m.gain = ch[m.from_]
        return p2

    def comp_terms(gains):
        """Gaze-compensation gains on the gaze joint, from the (scaled) body gains."""
        if jac is None:
            return {}
        g = GAZE[prof.name]
        comp = gaze_compensation(with_gains(gains), a.mode, jac, g["gaze_joint"], g["body_joints"])
        have = {t.from_ for t in prof.mapping(a.mode)[g["gaze_joint"]].terms()}
        missing = [c for c in comp if c not in have]
        if missing:
            print(f"!! {g['gaze_joint']} has no term for {missing}: add `- {{ from: <ch>, gain: 0 }}` lines by hand, then re-run")
        return {g["gaze_joint"]: {c: v for c, v in comp.items() if c in have}}

    # Fixed-point fit: the pipeline (soft limit, tracker) is not quite linear in
    # the gains, so re-measure the actual retargeted corpus and correct.
    vel_cap = None if a.vel_cap <= 0 else a.vel_cap
    props = propose_multipliers(prof, a.mode, vendor, cur_lin, a.min_mult, a.max_mult, a.headroom, skip, pipeline=before, vel_cap=vel_cap)
    mults = {p.joint: p.mult for p in props.values()}
    rows = [[p.joint, f"{p.vendor_abs_p95:.1f}", f"{p.current_abs_p95:.1f}", f"{p.raw_mult:.2f}", f"{p.mult:.2f}", p.cap_reason] for p in props.values()]
    print("\n# pass 1: proposed gain multipliers (vendor |.|p95 / current linear |.|p95)")
    print(fmt_table(rows, ["joint", "vendor |.|p95", "current |.|p95", "raw", "mult", "note"]))
    for it in range(2, a.passes + 1):
        trial = with_gains(scaled_gains(prof, a.mode, mults, comp_terms(scaled_gains(prof, a.mode, mults))))
        got = joint_stats(list(retarget_tables(humans, trial, a.mode).values()), joints, rest)
        rows = []
        for jn, p in props.items():
            v, s = vendor[jn], got[jn]
            k_amp = v.abs_p95 / max(s.abs_p95, 1e-9)
            k_vel = (vel_cap * v.vel_p95 / max(s.vel_p95, 1e-9)) if vel_cap else float("inf")
            k = min(k_amp, k_vel, 1.0 if p.mult >= a.max_mult else float("inf"))
            k = max(k, a.min_mult / p.mult)
            # never beyond the linear headroom cap already found in pass 1
            if "headroom" in p.cap_reason:
                k = min(k, 1.0)
            new = min(max(mults[jn] * k, a.min_mult), a.max_mult)
            rows.append([jn, f"{s.abs_p95:.1f}", f"{v.abs_p95:.1f}", f"{s.vel_p95:.0f}", f"{v.vel_p95:.0f}", f"{mults[jn]:.3f}", f"{new:.3f}"])
            mults[jn] = new
        print(f"\n# pass {it}: measured |.|p95 / vel p95 of the actual pipeline → corrected multipliers")
        print(fmt_table(rows, ["joint", "got p95", "vendor", "got vel", "vendor vel", "mult", "→ mult"]))

    fixed = comp_terms(scaled_gains(prof, a.mode, mults))
    if fixed:
        print("gaze-compensation gains: " + "; ".join(f"{jn}: " + ", ".join(f"{c}={v:+.3f}" for c, v in ch.items()) for jn, ch in fixed.items()))
    new_gains = scaled_gains(prof, a.mode, mults, fixed)
    old_gains = current_gains(prof, a.mode)
    rows = []
    for jn in joints:
        for ch, gnew in new_gains.get(jn, {}).items():
            gold = old_gains[jn][ch]
            if abs(gnew - gold) > 1e-9:
                rows.append([jn, ch, f"{gold:+.4g}", f"{gnew:+.4g}"])
    print("\n# gain changes")
    print(fmt_table(rows, ["joint", "channel", "old", "new"]) if rows else "(none)")

    text = open(prof.path, encoding="utf-8").read()
    changed = {jn: {ch: g for ch, g in terms.items() if abs(g - old_gains[jn][ch]) > 1e-9} for jn, terms in new_gains.items()}
    changed = {jn: ch for jn, ch in changed.items() if ch}
    new_text = rewrite_gains(text, a.mode, changed, a.stamp)
    tmp_path = prof.path + ".fit.tmp"
    open(tmp_path, "w", encoding="utf-8", newline="\n").write(new_text)
    fitted = load_profile(tmp_path)
    fitted.path = prof.path
    for jn, ch in new_gains.items():
        got = {t.from_: t.gain for t in fitted.mapping(a.mode)[jn].terms()}
        for c, v in ch.items():
            assert abs(got[c] - v) < 1e-6 or abs(got[c] - float(f"{v:.4g}")) < 1e-9, (jn, c, got[c], v)
    after = joint_stats(list(retarget_tables(humans, fitted, a.mode).values()), joints, rest)
    print(f"\n# AFTER: human clips through the fitted mapping (values − rest)")
    print(fmt_table(stats_rows(after, joints), HDR))
    rows = []
    for j in joints:
        if j in vendor and j in before and j in after and vendor[j].abs_p95 > 1e-9:
            rows.append([j, f"{vendor[j].abs_p95:.1f}", f"{before[j].abs_p95:.1f}", f"{before[j].abs_p95 / vendor[j].abs_p95:.2f}", f"{after[j].abs_p95:.1f}", f"{after[j].abs_p95 / vendor[j].abs_p95:.2f}"])
    print("\n# envelope match (|.|p95 ratio retargeted/vendor; 1.00 = matched)")
    print(fmt_table(rows, ["joint", "vendor", "before", "ratio", "after", "ratio"]))
    if prof.name in GAZE and not a.no_gaze:
        errs = gaze_error_cases(fitted, a.mode, urdf, LEAN_CASES, GAZE[prof.name]["gaze_joint"])
        print("gaze elevation error under lean, FITTED mapping: " + "; ".join(f"{ch}: {e:+.2f} deg" for ch, e, _ in errs))
    if a.write:
        os.replace(tmp_path, prof.path)
        print(f"\nwrote {prof.path} (stamp {a.stamp}); re-export web JSON: python -m animacy.cli profile export robots/{prof.name} -o web/robots/{prof.name}.json")
    else:
        os.remove(tmp_path)
        print("\n(dry run — pass --write to update ROBOT.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
