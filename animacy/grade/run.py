"""The end-to-end blind grading run and the pass rule.

    python scripts/grade_run.py --robots lamp reachy_mini --sources model retrieval envelope --seeds 2 \\
        --out data/grading/<timestamp>

Steps: probe (can the judge watch video?) -> clips (candidates + vendor
calibration) -> blind numbering + reels -> judge each reel in its own empty
workspace -> unseal -> ``results.json`` + ``docs/evidence/grading/<run>.md``.

THE GATE (``gate``): for each robot, the ``model`` source must score
overall >= 8.0 on ALL five movements, using the MEAN over seeds. Best-of-seeds
is reported for information and never used for the verdict. Vendor clips are
the calibration: if their mean overall is below 6 the rendering or the rubric
is broken and the run says so.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import kimi, rubric
from .movements import DEFAULT_CHECKPOINT, DETERMINISTIC_SOURCES, MOVEMENT_KEYS, VENDOR, build_clips
from .probe import run_probe
from .reel import MAX_REEL_SECONDS, Reel, plan_reels, render_reels, write_sealed_manifest
from .render import ROOT, ViewerRenderer, contact_sheet, joint_plot

GATE_SOURCE = "model"            # the default when web/models/model.json names no default_backend
GATE_THRESHOLD = 8.0
CALIBRATION_MIN = 6.0
GATE_RULE = "mean over seeds of the judge's overall score >= 8.0 on ALL five movements (best-of-seeds is never used)"
EVIDENCE_DIR = os.path.join(ROOT, "docs", "evidence", "grading")
PROVENANCE_FILES = ["robots/lamp/ROBOT.md", "robots/reachy_mini/ROBOT.md", "animacy/retarget.py", "web/models/model.json"]


# ---------------------------------------------------------------- provenance
def _sha1(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return ""


def provenance(checkpoint: str, run_dir: Optional[str] = None, robots: Sequence[str] = ()) -> Dict:
    """What exactly was graded: git HEAD, dirty files, hashes (+ copies under ``<run>/provenance``) of every
    file that decides the motion (ROBOT.md per robot, retarget.py, the web bundle contract, the checkpoint)."""
    files = list(PROVENANCE_FILES) + [f"robots/{r}/ROBOT.md" for r in robots if f"robots/{r}/ROBOT.md" not in PROVENANCE_FILES]
    ck = os.path.abspath(checkpoint)
    for name in ("model_info.json", "REPORT.md", "a2m.pt", "a2m_ar.pt", "vq.pt", "retrieval.json"):
        files.append(os.path.relpath(os.path.join(ck, name), ROOT).replace("\\", "/"))
    out = {"git_head": _git("rev-parse", "HEAD"), "git_head_short": _git("rev-parse", "--short", "HEAD"),
           "git_dirty": [l for l in _git("status", "--short").splitlines() if l.strip()],
           "checkpoint": ck, "files": {}}
    for rel in files:
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            continue
        st = os.stat(p)
        out["files"][rel] = {"sha1": _sha1(p), "mtime": dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                             "bytes": st.st_size}
        if run_dir and (rel.endswith(".md") or rel.endswith(".json") or rel.endswith(".py")):
            dst = os.path.join(run_dir, "provenance", rel.replace("/", "__"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(p, dst)
    return out


def resolve_gate_source(requested: str = "auto", sources: Sequence[str] = ()) -> Dict:
    """Which source the gate judges. ``auto`` = the SHIPPED default: ``default_backend`` in
    ``web/models/model.json`` (what a user of the web demo gets); else the named source."""
    bundle = os.path.join(ROOT, "web", "models", "model.json")
    shipped = None
    if os.path.isfile(bundle):
        try:
            shipped = json.load(open(bundle, encoding="utf-8")).get("default_backend")
        except (OSError, ValueError):
            shipped = None
    if requested == "auto":
        src = shipped or GATE_SOURCE
        why = "shipped default_backend in web/models/model.json" if shipped else "no default_backend in the bundle; fallback"
    else:
        src, why = requested, "requested on the command line"
    if sources and src not in sources:
        raise ValueError(f"gate source {src!r} ({why}) is not among the graded sources {list(sources)}")
    return {"source": src, "why": why, "shipped_default_backend": shipped}


# ---------------------------------------------------------------- the gate
def _mean(xs: Sequence[float]) -> Optional[float]:
    xs = [float(x) for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def gate(records: Sequence[Dict], robot: str, movements: Sequence[str] = MOVEMENT_KEYS, source: str = GATE_SOURCE,
         threshold: float = GATE_THRESHOLD) -> Dict:
    """The pass rule. ``records`` are unsealed per-clip results (see :func:`unseal`).

    A movement passes only if it has at least one scored clip and the MEAN of its ``overall`` scores
    over seeds is >= ``threshold``; the robot passes only if every movement passes."""
    per: Dict[str, Dict] = {}
    for mv in movements:
        vals = [r["overall"] for r in records
                if r["robot"] == robot and r["source"] == source and r["movement"] == mv and r.get("overall") is not None]
        m = _mean(vals)
        per[mv] = {"n": len(vals), "seeds": vals, "mean": m, "best": max(vals) if vals else None,
                   "pass": bool(vals) and m is not None and m >= threshold}
    return {"robot": robot, "source": source, "threshold": threshold, "rule": GATE_RULE,
            "per_movement": per, "pass": all(per[mv]["pass"] for mv in movements),
            "best_of_seeds_would_pass": all(per[mv]["best"] is not None and per[mv]["best"] >= threshold for mv in movements)}


def calibration(records: Sequence[Dict], robot: str, minimum: float = CALIBRATION_MIN) -> Dict:
    vals = [r["overall"] for r in records if r["robot"] == robot and r["source"] == VENDOR and r.get("overall") is not None]
    m = _mean(vals)
    return {"robot": robot, "n": len(vals), "mean": m, "minimum": minimum, "scores": vals,
            "ok": m is not None and m >= minimum,
            "note": None if (m is not None and m >= minimum) else
            "vendor clips average below the minimum: the rendering or the rubric is broken, candidate scores are not trustworthy"}


def consistency(records: Sequence[Dict], robot: str, source: str = "retrieval") -> Dict:
    """Retrieval is deterministic, so its two seeds are the SAME clip: the score gap is judge noise."""
    by_mv: Dict[str, List[float]] = {}
    for r in records:
        if r["robot"] == robot and r["source"] == source and r.get("overall") is not None:
            by_mv.setdefault(r["movement"], []).append(r["overall"])
    gaps = [max(v) - min(v) for v in by_mv.values() if len(v) >= 2]
    return {"robot": robot, "source": source, "pairs": len(gaps), "mean_abs_gap": _mean(gaps), "gaps": gaps}


# ---------------------------------------------------------------- judging
def judge_workspace_root(run_name: str) -> Path:
    """Where the judge's per-reel workspaces live: outside the repo, under a name that carries no project
    context (the reel path is quoted in the prompt, so the directory name is part of what the judge reads)."""
    root = Path(tempfile.gettempdir()) / "motion_judge" / run_name
    hits = rubric.forbidden_hits(str(root))
    if hits:
        raise RuntimeError(f"judge workspace path {root} contains forbidden words {hits}; set TMPDIR/TEMP elsewhere")
    return root


def prepare_workspace(reel_path: str, workspace: Path) -> Path:
    """An empty directory holding only the reel. Anything that was there is removed."""
    workspace = Path(workspace)
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    dst = workspace / os.path.basename(reel_path)
    shutil.copyfile(reel_path, dst)
    return dst


def check_workspace(workspace: Path) -> List[str]:
    """Files the judge could see. Raises if anything that is not a reel or a prompt file is there."""
    listing = kimi.workspace_listing(workspace)
    bad = [f for f in listing if not (f.endswith(".mp4") or f.endswith(".png") or f.startswith(".kimi-prompts/"))]
    leak = [f for f in listing if "manifest" in f.lower() or "result" in f.lower() or f.endswith(".json")]
    if bad or leak:
        raise RuntimeError(f"judge workspace {workspace} is not clean: {bad or leak}")
    return listing


def judge_reel(reel: Reel, workspace_root: Path, run_dir: str, timeout: int = 1800, attempts: int = 3,
               speech_strip: bool = True, log=print, dry_run: bool = False) -> Dict:
    """One reel -> the judge's validated JSON (cached in ``<run_dir>/kimi/<reel>.json``)."""
    cache_dir = os.path.join(run_dir, "kimi")
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, reel.name + ".json")
    if os.path.exists(cache):
        cached = json.load(open(cache, encoding="utf-8"))
        if cached.get("valid"):
            log(f"[judge] {reel.name}: cached")
            return cached
    ws = Path(workspace_root) / reel.name
    dst = prepare_workspace(reel.path, ws)
    listing = check_workspace(ws)
    prompt = rubric.build_prompt(str(dst.resolve()), reel.numbers, reel.robot, has_speech_strip=speech_strip)
    hits = rubric.forbidden_hits(prompt)
    if hits:
        raise RuntimeError(f"rubric prompt contains forbidden words {hits}; refusing to run")
    rec: Dict = {"reel": reel.name, "robot": reel.robot, "numbers": reel.numbers, "workspace": str(ws),
                 "workspace_listing_before": listing, "attempts": [], "valid": False, "answer": None}
    if dry_run:
        rec["dry_run"] = True
        return rec
    for k in range(attempts):
        t0 = time.perf_counter()
        try:
            ans = kimi.ask_json(prompt, ws, timeout=timeout)
        except kimi.KimiError as e:
            rec["attempts"].append({"error": str(e)[:500], "seconds": time.perf_counter() - t0})
            log(f"[judge] {reel.name}: attempt {k + 1} failed: {str(e)[:200]}")
            continue
        errs = rubric.validate_response(ans, reel.numbers)
        rec["attempts"].append({"seconds": ans.get("_seconds"), "errors": errs, "raw": ans.get("_raw", "")[:20000]})
        if not errs:
            rec["answer"] = {k2: v for k2, v in ans.items() if not k2.startswith("_")}
            rec["valid"] = True
            log(f"[judge] {reel.name}: ok in {ans.get('_seconds', 0):.0f}s")
            break
        log(f"[judge] {reel.name}: attempt {k + 1} malformed: {errs[:3]}")
    rec["workspace_listing_after"] = kimi.workspace_listing(ws)
    with open(cache, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def judge_all(plans: Dict[str, List[Reel]], workspace_root: Path, run_dir: str, timeout: int = 1800,
              parallel: int = 2, speech_strip: bool = True, log=print, dry_run: bool = False) -> List[Dict]:
    reels = [r for rs in plans.values() for r in rs]
    if parallel <= 1 or len(reels) <= 1:
        return [judge_reel(r, workspace_root, run_dir, timeout, speech_strip=speech_strip, log=log, dry_run=dry_run)
                for r in reels]
    with ThreadPoolExecutor(max_workers=min(parallel, len(reels))) as pool:
        futs = [pool.submit(judge_reel, r, workspace_root, run_dir, timeout, 3, speech_strip, log, dry_run) for r in reels]
        return [f.result() for f in futs]


# ---------------------------------------------------------------- unsealing + reports
def unseal(judgements: Sequence[Dict], manifest: Dict) -> List[Dict]:
    """Join the judge's numbered scores with the sealed manifest -> one record per clip."""
    records: List[Dict] = []
    for j in judgements:
        robot = j["robot"]
        clips = manifest["robots"][robot]["clips"]
        answers = {int(c["clip"]): c for c in ((j.get("answer") or {}).get("clips") or []) if "clip" in c}
        for n in j["numbers"]:
            pub = clips[str(n)]
            a = answers.get(n)
            scores = rubric.normalise_scores(a) if a else {}
            records.append({**pub, "number": n, "reel": j["reel"], "scores": scores,
                            "overall": scores.get("overall"), "description": rubric.one_line(a.get("description", "")) if a else "",
                            "reason": rubric.one_line(a.get("reason", "")) if a else ""})
    return records


def summarise(records: Sequence[Dict], robots: Sequence[str], sources: Sequence[str],
              gate_source: str = GATE_SOURCE) -> Dict:
    """Per robot x source x movement: mean/per-seed overall + the gate (on ``gate_source``), the same rule
    applied to every other source for information, calibration and consistency."""
    out: Dict = {"robots": {}, "gate_source": gate_source}
    for robot in robots:
        table: Dict[str, Dict[str, Dict]] = {}
        for mv in MOVEMENT_KEYS:
            table[mv] = {}
            for src in list(sources) + [VENDOR]:
                vals = [r["overall"] for r in records if r["robot"] == robot and r["source"] == src and r["movement"] == mv
                        and r.get("overall") is not None]
                dims = {d: _mean([r["scores"].get(d) for r in records if r["robot"] == robot and r["source"] == src
                                  and r["movement"] == mv and r["scores"].get(d) is not None]) for d in rubric.DIMENSIONS}
                table[mv][src] = {"mean": _mean(vals), "seeds": vals, "dims": dims}
        out["robots"][robot] = {"table": table, "gate": gate(records, robot, source=gate_source),
                                "gate_by_source": {src: gate(records, robot, source=src) for src in sources},
                                "calibration": calibration(records, robot), "consistency": consistency(records, robot)}
    return out


def _fmt(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.1f}"


def compare_runs(base: Dict, new: Dict) -> Dict:
    """Side by side, per robot x source x movement: baseline mean -> new mean (delta), gates and calibration."""
    out: Dict = {"baseline": base["run"], "new": new["run"], "robots": {}}
    for robot in new["summary"]["robots"]:
        if robot not in base["summary"]["robots"]:
            continue
        bt, nt = base["summary"]["robots"][robot]["table"], new["summary"]["robots"][robot]["table"]
        cells: Dict[str, Dict[str, Dict]] = {}
        for mv in MOVEMENT_KEYS:
            cells[mv] = {}
            for src in set(bt.get(mv, {})) | set(nt.get(mv, {})):
                b = (bt.get(mv, {}).get(src) or {}).get("mean")
                n = (nt.get(mv, {}).get(src) or {}).get("mean")
                cells[mv][src] = {"baseline": b, "new": n, "delta": (n - b) if (b is not None and n is not None) else None}
        out["robots"][robot] = {
            "table": cells,
            "gate": {"baseline": base["gate"][robot]["pass"], "new": new["gate"][robot]["pass"]},
            "calibration": {"baseline": base["calibration"][robot]["mean"], "new": new["calibration"][robot]["mean"]},
            "model_min": {"baseline": min([v["mean"] for v in base["gate"][robot]["per_movement"].values() if v["mean"] is not None], default=None),
                          "new": min([v["mean"] for v in new["gate"][robot]["per_movement"].values() if v["mean"] is not None], default=None)},
        }
    return out


def _comparison_markdown(cmp: Dict, sources: Sequence[str]) -> List[str]:
    L = [f"## Compared with baseline `{cmp['baseline']}`\n",
         "Overall score, mean over seeds: baseline -> this run (delta). Same movements, rubric, blind protocol and judge.\n"]
    for robot, c in cmp["robots"].items():
        srcs = [s for s in list(sources) + [VENDOR] if any(s in c["table"][mv] for mv in MOVEMENT_KEYS)]
        L.append(f"### {robot}\n")
        L.append("| movement | " + " | ".join(srcs) + " |")
        L.append("|---|" + "---|" * len(srcs))
        for mv in MOVEMENT_KEYS:
            row = []
            for src in srcs:
                x = c["table"][mv].get(src)
                if not x or (x["baseline"] is None and x["new"] is None):
                    row.append("-")
                else:
                    d = "" if x["delta"] is None else f" ({x['delta']:+.1f})"
                    row.append(f"{_fmt(x['baseline'])} -> {_fmt(x['new'])}{d}")
            L.append(f"| {mv} | " + " | ".join(row) + " |")
        g, cal, mm = c["gate"], c["calibration"], c["model_min"]
        L.append("")
        L.append(f"Gate: baseline **{'PASS' if g['baseline'] else 'FAIL'}** (min model movement {_fmt(mm['baseline'])}) -> "
                 f"this run **{'PASS' if g['new'] else 'FAIL'}** (min {_fmt(mm['new'])}). Vendor calibration "
                 f"{_fmt(cal['baseline'])} -> {_fmt(cal['new'])}.\n")
    return L


def _provenance_markdown(prov: Optional[Dict]) -> List[str]:
    if not prov:
        return []
    L = ["## Provenance\n",
         f"git HEAD `{prov.get('git_head_short') or prov.get('git_head')}`"
         + (f", {len(prov['git_dirty'])} uncommitted change(s) in the tree" if prov.get("git_dirty") else ", clean tree")
         + f". Checkpoint `{prov.get('checkpoint')}`.\n",
         "| file | sha1 | modified |", "|---|---|---|"]
    for rel, f in prov.get("files", {}).items():
        L.append(f"| `{rel}` | `{f['sha1'][:12]}` | {f['mtime']} |")
    L.append("")
    return L


def markdown_report(run_name: str, meta: Dict, summary: Dict, records: Sequence[Dict], judgements: Sequence[Dict],
                    probe: Dict, comparison: Optional[Dict] = None) -> str:
    L: List[str] = []
    L.append(f"# Blind motion grading: `{run_name}`\n")
    if meta.get("label"):
        L.append(f"**{meta['label']}**\n")
    L.append(f"Generated {meta['generated']} by `{meta['command']}`. Judge: `{meta['judge_model']}` via the local kimi CLI, "
             f"{meta['kimi_calls']} call(s). Checkpoint: `{meta['checkpoint']}`. Seed {meta['seed']}"
             + (f"; seeds per source {meta['seeds_by_source']}" if meta.get("seeds_by_source") else "") + ".\n")
    L.extend(_provenance_markdown(meta.get("provenance")))
    L.append("## What the judge could see\n")
    pa = probe.get("answer") or {}
    if probe.get("video_seen"):
        L.append(f"- Video: **yes**. In the 3 s probe it read the title card (`{pa.get('title_card_text', '')!s}`) and described "
                 f"the motion correctly (`{rubric.one_line(pa.get('what_moved', ''), 160)}`).")
    else:
        L.append(f"- Video: **NO** (probe failed: {probe.get('error') or pa}). The run fell back to contact sheets + joint plots.")
    L.append(f"- Frames it reported seeing in the 3 s probe: {pa.get('frames_seen')} (`{rubric.one_line(pa.get('notes', ''), 220)}`).")
    L.append(f"- Audio: reported has_audio={pa.get('has_audio')}, but it cannot listen to it (probe notes). "
             "**The judge grades from video only.** The `timing` dimension therefore measures rhythm plausibility "
             "against the transcript card and the burned-in loudness strip, not audio sync; the pass rule uses "
             "`overall` only.\n")
    L.append("## Gate\n")
    gs = summary.get("gate_source", GATE_SOURCE)
    gsi = meta.get("gate_source_info") or {}
    L.append(f"Rule: {GATE_RULE}. **Source under test: `{gs}`** ({gsi.get('why', 'the source named when the run was made')}).\n")
    cal_line = "; ".join(f"{robot} {_fmt(s['calibration']['mean'])} over {s['calibration']['n']} clips"
                         for robot, s in summary["robots"].items())
    L.append(f"**The vendor's own hand-authored clips score {cal_line} on this rubric.** That is what a shipped, "
             f"hand-made clip earns from this judge; 8.0 is above it.\n")
    L.append("| robot | " + " | ".join(MOVEMENT_KEYS) + " | min | verdict | best-of-seeds (info only) |")
    L.append("|---|" + "---|" * (len(MOVEMENT_KEYS) + 3))
    for robot, s in summary["robots"].items():
        g = s["gate"]
        means = [g["per_movement"][mv]["mean"] for mv in MOVEMENT_KEYS]
        mn = min([m for m in means if m is not None], default=None)
        L.append(f"| {robot} | " + " | ".join(_fmt(m) for m in means) + f" | {_fmt(mn)} | **{'PASS' if g['pass'] else 'FAIL'}** | "
                 f"{'would pass' if g['best_of_seeds_would_pass'] else 'would still fail'} |")
    L.append("")
    if any("gate_by_source" in s for s in summary["robots"].values()):
        L.append("The same rule applied to every source (information only; only the source under test decides):\n")
        L.append("| robot | source | " + " | ".join(MOVEMENT_KEYS) + " | min | would |")
        L.append("|---|---|" + "---|" * (len(MOVEMENT_KEYS) + 2))
        for robot, s in summary["robots"].items():
            for src, g2 in (s.get("gate_by_source") or {}).items():
                means2 = [g2["per_movement"][mv]["mean"] for mv in MOVEMENT_KEYS]
                mn2 = min([m for m in means2 if m is not None], default=None)
                L.append(f"| {robot} | {src}{' (under test)' if src == gs else ''} | " + " | ".join(_fmt(m) for m in means2)
                         + f" | {_fmt(mn2)} | {'PASS' if g2['pass'] else 'FAIL'} |")
        L.append("")
    L.append("## Overall score by robot x source x movement (mean over seeds; per-seed in brackets)\n")
    for robot, s in summary["robots"].items():
        srcs = list(meta["sources"]) + [VENDOR]
        L.append(f"### {robot}\n")
        L.append("| movement | " + " | ".join(srcs) + " |")
        L.append("|---|" + "---|" * len(srcs))
        for mv in MOVEMENT_KEYS:
            cells = []
            for src in srcs:
                cell = s["table"][mv][src]
                cells.append(f"{_fmt(cell['mean'])} [{', '.join(_fmt(v) for v in cell['seeds'])}]" if cell["seeds"] else "-")
            L.append(f"| {mv} | " + " | ".join(cells) + " |")
        cal = s["calibration"]
        L.append("")
        L.append(f"Calibration (vendor clips): mean overall **{_fmt(cal['mean'])}** over {cal['n']} clips "
                 f"(minimum {cal['minimum']}): {'OK' if cal['ok'] else 'BROKEN - ' + str(cal['note'])}.")
        con = s["consistency"]
        if con["pairs"]:
            L.append(f"Judge self-consistency: the two `retrieval` seeds are identical clips; mean |overall gap| = "
                     f"**{_fmt(con['mean_abs_gap'])}** over {con['pairs']} pairs (gaps {con['gaps']}).")
        L.append("")
        L.append("Dimension means by source (over all movements and seeds):\n")
        L.append("| source | " + " | ".join(rubric.SCORE_KEYS) + " |")
        L.append("|---|" + "---|" * len(rubric.SCORE_KEYS))
        for src in srcs:
            cells = [_fmt(_mean([r["scores"].get(d) for r in records if r["robot"] == robot and r["source"] == src
                                 and r["scores"].get(d) is not None])) for d in rubric.SCORE_KEYS]
            L.append(f"| {src} | " + " | ".join(cells) + " |")
        L.append("")
    if comparison:
        L.extend(_comparison_markdown(comparison, meta["sources"]))
    L.append("## Every clip, unsealed\n")
    L.append("| robot | # | origin | overall | lifelike | intent | timing | physical | appeal | what the judge saw |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(records, key=lambda r: (r["robot"], r["number"])):
        sc = r["scores"]
        L.append(f"| {r['robot']} | {r['number']} | `{r['id']}` | {_fmt(r.get('overall'))} | " +
                 " | ".join(_fmt(sc.get(d)) for d in rubric.DIMENSIONS) + f" | {r['description'].replace('|', '/')} |")
    L.append("")
    L.append("## Judge calls and the judge's own notes (verbatim)\n")
    for j in judgements:
        att = j.get("attempts") or []
        secs = sum((a.get("seconds") or 0) for a in att)
        after = j.get("workspace_listing_after") or []
        own = [f for f in after if f not in (j.get("workspace_listing_before") or [])]
        L.append(f"- `{j['reel']}` clips {j['numbers'][0]}..{j['numbers'][-1]}: {'valid' if j.get('valid') else 'INVALID'} after "
                 f"{len(att)} attempt(s), {secs:.0f} s. Workspace held before the call: {j.get('workspace_listing_before')}"
                 + (f"; the judge left {len(own)} file(s) of its own analysis (frames, sheets, motion tables)." if own else "."))
        notes = str((j.get("answer") or {}).get("notes") or "").strip()
        if notes:
            for line in notes.splitlines():
                L.append(f"  > {line}")
        L.append("")
    if meta.get("notes"):
        L.append("## Notes\n")
        for n in meta["notes"]:
            L.append(f"- {n}")
        L.append("")
    L.append("## Limitations\n")
    L.append("- The judge samples frames from the video (it reported ~2 per second in the probe); micro-motion, overshoot/settle "
             "and jitter are only partly visible to it, so `lifelike`/`physical` are judged from sampled poses.")
    L.append("- The judge cannot hear the audio; `timing` is judged from the loudness strip burned into the frames "
             "(rhythm plausibility, not measured audio sync).")
    L.append("- At ~2 samples per second a 0.5 s event (a brow flick, an antenna snap) can fall between the judge's "
             "samples entirely; `--speed 0.5` renders a separate slow-motion variant for that question and is never "
             "mixed into a gate run.")
    L.append("- Vendor clips are silent and carry an 'expresses' card instead of a 'says' card; the judge could in principle "
             "notice that difference, which is why they are reported as calibration, not compared head-to-head.")
    L.append("- Lamp joint values are the vendor's servo units (labelled degrees, within ~10%); see robots/lamp/ROBOT.md.")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main
def run(robots: Sequence[str], sources: Sequence[str], seeds: int, out_dir: str, checkpoint: str = DEFAULT_CHECKPOINT,
        seed: int = 0, max_reel_seconds: float = MAX_REEL_SECONDS, kimi_timeout: int = 1800, parallel: int = 2,
        gpu: bool = True, speech_strip: bool = True, no_kimi: bool = False, force_probe: bool = False,
        tts_engine: str = "auto", evidence_dir: str = EVIDENCE_DIR, zoom: float = 0.9, speed: float = 1.0,
        seeds_deterministic: Optional[int] = None, label: str = "", compare: Optional[str] = None,
        gate_source: str = "auto", log=print) -> Dict:
    t_start = time.perf_counter()
    run_dir = os.path.abspath(out_dir)
    os.makedirs(run_dir, exist_ok=True)
    run_name = os.path.basename(run_dir.rstrip("/\\"))
    workspace_root = judge_workspace_root(run_name)
    command = "python scripts/grade_run.py " + " ".join(sys.argv[1:]) if sys.argv and "grade_run" in sys.argv[0] else "animacy.grade.run.run(...)"
    seeds_by_source = {s: seeds_deterministic for s in sources if s in DETERMINISTIC_SOURCES} if seeds_deterministic else {}
    gsi = resolve_gate_source(gate_source, sources)
    meta = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "command": command, "robots": list(robots),
            "sources": list(sources), "seeds": seeds, "seeds_by_source": seeds_by_source, "checkpoint": checkpoint,
            "seed": seed, "judge_model": kimi.DEFAULT_MODEL, "workspace_root": str(workspace_root),
            "speech_strip": speech_strip, "gpu": gpu, "speed": speed, "label": label, "compare": compare, "kimi_calls": 0,
            "gate_source": gsi["source"], "gate_source_info": gsi, "provenance": provenance(checkpoint, run_dir, robots)}
    log(f"[run] gate source: {gsi['source']} ({gsi['why']})")
    log(f"[run] provenance: git {meta['provenance']['git_head_short']}, "
        f"{len(meta['provenance']['git_dirty'])} dirty file(s), checkpoint {checkpoint}")
    if speed != 1.0:
        log(f"[run] speed {speed}: slow-motion VARIANT (cards say so); never compare its gate with a speed-1 run")
    log(f"[run] {run_name}: robots={list(robots)} sources={list(sources)} seeds={seeds} -> {run_dir}")

    with ViewerRenderer(gpu=gpu, zoom=zoom) as renderer:
        # 1. probe: can the judge watch video?
        if no_kimi:
            probe = {"skipped": True, "video_seen": True}
        else:
            probe = run_probe(run_dir, renderer, workspace_root, force=force_probe)
            meta["kimi_calls"] += 1
            log(f"[probe] video_seen={probe.get('video_seen')} frames_seen={probe.get('frames_seen')} "
                f"card_ok={probe.get('card_ok')} motion_ok={probe.get('motion_ok')}")
        video_ok = bool(probe.get("video_seen"))

        # 2. clips
        clips = build_clips(robots, sources, seeds, run_dir, checkpoint=checkpoint, tts_engine=tts_engine,
                            seeds_by_source=seeds_by_source, log=log)

        # 3. blind plan + reels
        plans = plan_reels(clips, seed, max_reel_seconds)
        t0 = time.perf_counter()
        render_reels(plans, renderer, run_dir, speech_strip=speech_strip, speed=speed, log=log)
        meta["render_seconds"] = round(time.perf_counter() - t0, 1)
        if not video_ok:
            fallback_sheets(plans, renderer, run_dir, log=log)
        manifest_path = write_sealed_manifest(plans, run_dir, seed, extra={"meta": meta})
        manifest = json.load(open(manifest_path, encoding="utf-8"))
        log(f"[reel] sealed manifest -> {manifest_path}")

    # 4. judge (renderer closed: the browser is not needed while Kimi works)
    if not video_ok:
        judgements = judge_all_sheets(plans, workspace_root, run_dir, kimi_timeout, parallel, log=log, dry_run=no_kimi)
    else:
        judgements = judge_all(plans, workspace_root, run_dir, kimi_timeout, parallel, speech_strip, log=log, dry_run=no_kimi)
    meta["kimi_calls"] += sum(len(j.get("attempts") or []) for j in judgements)

    # 5. unseal + reports
    records = unseal(judgements, manifest)
    summary = summarise(records, robots, sources, gate_source=gsi["source"])
    meta["total_seconds"] = round(time.perf_counter() - t_start, 1)
    results = {"schema": "animacy.grading.results.v1", "run": run_name, "meta": meta, "probe": probe,
               "gate": {r: summary["robots"][r]["gate"] for r in robots},
               "calibration": {r: summary["robots"][r]["calibration"] for r in robots},
               "consistency": {r: summary["robots"][r]["consistency"] for r in robots},
               "summary": summary, "records": records, "judgements": judgements}
    comparison = None
    if compare and not no_kimi:
        base_path = os.path.join(os.path.abspath(compare), "results.json")
        if os.path.exists(base_path):
            comparison = compare_runs(json.load(open(base_path, encoding="utf-8")), results)
            results["comparison"] = comparison
        else:
            log(f"[compare] no results.json in {compare}; skipping the comparison")
    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    if not no_kimi:
        os.makedirs(evidence_dir, exist_ok=True)
        md = markdown_report(run_name, meta, summary, records, judgements, probe, comparison)
        with open(os.path.join(evidence_dir, run_name + ".md"), "w", encoding="utf-8") as fh:
            fh.write(md)
        log(f"[report] docs/evidence/grading/{run_name}.md")
    for robot in robots:
        g = summary["robots"][robot]["gate"]
        cal = summary["robots"][robot]["calibration"]
        means = ", ".join(f"{mv}={_fmt(g['per_movement'][mv]['mean'])}" for mv in MOVEMENT_KEYS)
        log(f"[gate] {robot}: {'PASS' if g['pass'] else 'FAIL'} ({gsi['source']} mean over seeds: {means}; "
            f"vendor calibration {_fmt(cal['mean'])}{'' if cal['ok'] else ' BROKEN'})")
    return results


def add_note(run_dir: str, note: str) -> None:
    """Append a note to a finished run's results (shown in its report)."""
    rpath = os.path.join(os.path.abspath(run_dir), "results.json")
    results = json.load(open(rpath, encoding="utf-8"))
    results["meta"].setdefault("notes", []).append(note)
    with open(rpath, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)


def report_from(run_dir: str, evidence_dir: str = EVIDENCE_DIR, label: Optional[str] = None,
                compare: Optional[str] = None) -> str:
    """Regenerate the markdown summary from an existing ``results.json`` (report code can improve after a run).
    ``label`` is stored back into the run's results so it sticks; ``compare`` adds/refreshes the baseline comparison."""
    run_dir = os.path.abspath(run_dir)
    rpath = os.path.join(run_dir, "results.json")
    results = json.load(open(rpath, encoding="utf-8"))
    run_name = results["run"]
    changed = False
    if label is not None:
        results["meta"]["label"] = label
        changed = True
    if compare:
        base = json.load(open(os.path.join(os.path.abspath(compare), "results.json"), encoding="utf-8"))
        results["comparison"] = compare_runs(base, results)
        results["meta"]["compare"] = compare
        changed = True
    if changed:
        with open(rpath, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)
    md = markdown_report(run_name, results["meta"], results["summary"], results["records"], results["judgements"],
                         results["probe"], results.get("comparison"))
    os.makedirs(evidence_dir, exist_ok=True)
    path = os.path.join(evidence_dir, run_name + ".md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(md)
    return path


# ---------------------------------------------------------------- fallback (judge cannot watch video)
def fallback_sheets(plans: Dict[str, List[Reel]], renderer: ViewerRenderer, run_dir: str, log=print) -> None:
    """Contact sheets + joint plots per clip, for a judge that cannot watch video."""
    from ..profile import find_robot

    d = os.path.join(run_dir, "sheets")
    os.makedirs(d, exist_ok=True)
    for robot, reels in plans.items():
        prof = find_robot(robot)
        for r in reels:
            for e in r.entries:
                frames = renderer.render_frames(robot, e.clip.table, prof)
                contact_sheet(frames, os.path.join(d, f"{robot}_clip{e.number:03d}_sheet.png"))
                joint_plot(e.clip.table, prof, os.path.join(d, f"{robot}_clip{e.number:03d}_joints.png"), f"Clip {e.number}")
    log(f"[fallback] contact sheets + joint plots -> {d}")


def judge_all_sheets(plans: Dict[str, List[Reel]], workspace_root: Path, run_dir: str, timeout: int, parallel: int,
                     log=print, dry_run: bool = False) -> List[Dict]:
    """Same protocol as :func:`judge_all` with PNG sheets in place of the reel."""
    out: List[Dict] = []
    sheets = os.path.join(run_dir, "sheets")
    for robot, reels in plans.items():
        for r in reels:
            ws = Path(workspace_root) / r.name
            if ws.exists():
                shutil.rmtree(ws)
            ws.mkdir(parents=True)
            files = []
            for e in r.entries:
                for suffix in ("sheet", "joints"):
                    src = os.path.join(sheets, f"{robot}_clip{e.number:03d}_{suffix}.png")
                    shutil.copyfile(src, ws / os.path.basename(src))
                files.append((e.number, e.clip.card_line))
            listing = check_workspace(ws)
            intro = "\n".join(f"Clip {n}: {card}: files {robot}_clip{n:03d}_sheet.png (12 frames, left-to-right then "
                              f"top-to-bottom, evenly spaced over the clip) and {robot}_clip{n:03d}_joints.png (each joint against time)"
                              for n, card in files)
            prompt = rubric.build_prompt(str(ws.resolve()), r.numbers, robot, has_speech_strip=False)
            prompt = prompt.replace(f"Watch the ENTIRE video file at {ws.resolve()}.",
                                    f"You cannot watch video here; instead look at every image in {ws.resolve()}:\n{intro}\n")
            if rubric.forbidden_hits(prompt):
                raise RuntimeError("fallback prompt contains forbidden words")
            rec: Dict = {"reel": r.name, "robot": robot, "numbers": r.numbers, "workspace": str(ws), "fallback": "sheets",
                         "workspace_listing_before": listing, "attempts": [], "valid": False, "answer": None}
            if not dry_run:
                for k in range(3):
                    try:
                        ans = kimi.ask_json(prompt, ws, timeout=timeout)
                    except kimi.KimiError as e:
                        rec["attempts"].append({"error": str(e)[:500]})
                        continue
                    errs = rubric.validate_response(ans, r.numbers)
                    rec["attempts"].append({"seconds": ans.get("_seconds"), "errors": errs})
                    if not errs:
                        rec["answer"] = {k2: v for k2, v in ans.items() if not k2.startswith("_")}
                        rec["valid"] = True
                        break
            out.append(rec)
            log(f"[judge/sheets] {r.name}: {'valid' if rec['valid'] else 'invalid'}")
    return out


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Blind motion grading with Kimi K3 (the animacy acceptance gate).")
    ap.add_argument("--robots", nargs="+", default=["lamp", "reachy_mini"])
    ap.add_argument("--sources", nargs="+", default=["model", "retrieval", "envelope"])
    ap.add_argument("--seeds", type=int, default=2, help="seeds for the stochastic sources (model)")
    ap.add_argument("--seeds-deterministic", type=int, default=1,
                    help=f"seeds for the deterministic sources {DETERMINISTIC_SOURCES} (0 = same as --seeds)")
    ap.add_argument("--label", default="", help="one-line label for the report (e.g. 'baseline: pre-fit mapping, v1')")
    ap.add_argument("--compare", default=None, metavar="RUN_DIR", help="baseline run to compare with, side by side")
    ap.add_argument("--gate-source", default="auto",
                    help="source the gate judges: auto = the shipped default_backend in web/models/model.json; or a name")
    ap.add_argument("--slow-variant", action="store_true",
                    help="after the run, render + judge the same clips at 0.5x as a SEPARATE sub-run (<out>_slow); never gated")
    ap.add_argument("--out", default=None, help="run directory (default data/grading/<timestamp>)")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--seed", type=int, default=0, help="shuffle seed for the blind order")
    ap.add_argument("--max-reel-seconds", type=float, default=MAX_REEL_SECONDS)
    ap.add_argument("--kimi-timeout", type=int, default=1800)
    ap.add_argument("--parallel", type=int, default=2, help="reels judged concurrently")
    ap.add_argument("--software", action="store_true", help="render with SwiftShader instead of the GPU (slow)")
    ap.add_argument("--no-speech-strip", action="store_true", help="do not burn the loudness strip into the frames")
    ap.add_argument("--no-kimi", action="store_true", help="build clips, reels and the sealed manifest only")
    ap.add_argument("--force-probe", action="store_true", help="re-run the video probe even if cached")
    ap.add_argument("--tts", default="auto", help="animacy.tts engine (auto|sapi|espeak|kokoro)")
    ap.add_argument("--zoom", type=float, default=0.9, help="camera distance multiplier (smaller = closer)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback speed of the rendered clips; 0.5 = slow-motion variant (card says so), for a SEPARATE run only")
    ap.add_argument("--report-only", default=None, metavar="RUN_DIR",
                    help="regenerate docs/evidence/grading/<run>.md from an existing results.json and exit "
                         "(--label and --compare apply and are stored)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    if a.report_only:
        print("[report]", report_from(a.report_only, label=a.label or None, compare=a.compare))
        return 0
    out = a.out or os.path.join(ROOT, "data", "grading", dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    common = dict(checkpoint=a.checkpoint, seed=a.seed, max_reel_seconds=a.max_reel_seconds, kimi_timeout=a.kimi_timeout,
                  parallel=a.parallel, gpu=not a.software, speech_strip=not a.no_speech_strip, no_kimi=a.no_kimi,
                  force_probe=a.force_probe, tts_engine=a.tts, zoom=a.zoom,
                  seeds_deterministic=(a.seeds_deterministic or None), gate_source=a.gate_source)
    results = run(a.robots, a.sources, a.seeds, out, speed=a.speed, label=a.label, compare=a.compare, **common)
    verdict = 0 if all(results["gate"][r]["pass"] for r in a.robots) else 1
    if a.slow_variant and not a.no_kimi:
        slow_out = out.rstrip("/\\") + "_slow"
        run_name = os.path.basename(out.rstrip("/\\"))
        run(a.robots, a.sources, a.seeds, slow_out, speed=0.5,
            label=f"slow-motion sub-run (0.5x) of {run_name}: same clips, cards say 'slow motion'; NOT a gate",
            compare=out, **common)
        add_note(out, f"Slow-motion sub-run (0.5x, not a gate): docs/evidence/grading/{os.path.basename(slow_out)}.md")
        report_from(out)
    if a.no_kimi:
        return 0
    return verdict


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
