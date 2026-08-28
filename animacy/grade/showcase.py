"""Export a run's best-scoring clips per movement as clean MP4s (for a demo video).

    python scripts/grade_showcase.py --run data/grading/<run> --robot lamp --source retrieval \\
        --out data/grading/showcase_lamp

Clean = re-rendered from the run's saved joint table with the utterance audio,
no blind title card and no loudness strip. Scores and the judge's description
go to ``showcase.json`` next to the videos.

Sealing: a held-out clip's audio IS the sealed utterance. Such clips are written
under ``SEALED_heldout/`` with a README, never mixed with the publishable ones;
publishing one burns the gate's held-out set.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..profile import find_robot
from .movements import HELDOUT_SET, MOVEMENT_KEYS, VENDOR, _wav_cache_path
from .render import ViewerRenderer
from .run import _set, redact_lines, sealed_lines

SEALED_DIR = "SEALED_heldout"
SEALED_README = """These clips speak the grader's SEALED held-out lines. They exist here only because they scored best.
Do NOT publish, post, or paste them anywhere: doing so tells every fixer the held-out utterances and burns
the gate (the sealed set would have to be replaced). Use the clips in the parent folder for the demo video.
"""


def pick_best(records: Sequence[Dict], robot: str, source: str, movements: Sequence[str] = MOVEMENT_KEYS,
              line_sets: Optional[Sequence[str]] = None) -> Dict[str, List[Dict]]:
    """Per movement: the robot's clips of ``source`` sorted best first (overall, then lifelike, then appeal,
    then lower seed). ``line_sets`` restricts to tuning/heldout; None = both."""
    out: Dict[str, List[Dict]] = {}
    for mv in movements:
        mine = [r for r in records if r["robot"] == robot and r["source"] == source and r["movement"] == mv
                and r.get("overall") is not None and (line_sets is None or _set(r) in line_sets)]
        # ties go to publishable (tuning) clips: a sealed clip is useless for a demo
        mine.sort(key=lambda r: (-r["overall"], -r["scores"].get("lifelike", 0), -r["scores"].get("appeal", 0),
                                 1 if _set(r) == HELDOUT_SET else 0, r.get("seed") if r.get("seed") is not None else 0))
        out[mv] = mine
    return out


def _load_table(run_dir: str, clip_id: str) -> pd.DataFrame:
    p = os.path.join(run_dir, "clips", clip_id.replace("/", "__") + ".json")
    obj = json.load(open(p, encoding="utf-8"))
    t = np.asarray(obj["t"], dtype=np.float64)
    return pd.DataFrame({"t": t - t[0], **{k: np.asarray(v, dtype=np.float64) for k, v in obj["data"].items()}})


def _load_audio(run_dir: str, card_line: str, engine: str = "auto"):
    import soundfile as sf

    if not card_line.startswith('The robot says: "'):
        return None, 16000
    text = card_line[len('The robot says: "'):-1]
    path = _wav_cache_path(os.path.join(run_dir, "tts"), text, engine)
    if not os.path.exists(path):
        return None, 16000
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return data.mean(axis=1), sr


def export_showcase(run_dir: str, robot: str, out_dir: str, source: str = "retrieval", per_movement: int = 1,
                    include_sealed: bool = True, gpu: bool = True, zoom: float = 0.9, log=print) -> Dict:
    """Render the best ``per_movement`` clips of ``source`` per movement to ``out_dir`` and write showcase.json."""
    run_dir = os.path.abspath(run_dir)
    results = json.load(open(os.path.join(run_dir, "results.json"), encoding="utf-8"))
    records = results["records"]
    sealed = sealed_lines(records)
    best = pick_best(records, robot, source)
    profile = find_robot(robot)
    os.makedirs(out_dir, exist_ok=True)
    manifest: Dict = {"schema": "animacy.grading.showcase.v1", "run": results["run"], "robot": robot, "source": source,
                      "label": results["meta"].get("label"), "note": "clean re-render (no card, no strip) with the utterance audio",
                      "movements": {}}
    with ViewerRenderer(gpu=gpu, zoom=zoom, log=log) as renderer:
        for mv, ranked in best.items():
            entries = []
            chosen = list(ranked[:per_movement])
            # a sealed winner always gets the best publishable clip exported next to it
            if any(_set(r) == HELDOUT_SET for r in chosen) and not any(_set(r) != HELDOUT_SET for r in chosen):
                pub = next((r for r in ranked if _set(r) != HELDOUT_SET), None)
                if pub is not None:
                    chosen.append(pub)
            for rank, r in enumerate(chosen):
                is_sealed = _set(r) == HELDOUT_SET
                if is_sealed and not include_sealed:
                    continue
                sub = os.path.join(out_dir, SEALED_DIR) if is_sealed else out_dir
                os.makedirs(sub, exist_ok=True)
                if is_sealed:
                    with open(os.path.join(sub, "README.txt"), "w", encoding="utf-8") as fh:
                        fh.write(SEALED_README)
                name = f"{robot}_{mv}_{source}{'_heldout' if is_sealed else ''}_s{r.get('seed', 0)}_score{r['overall']:g}.mp4"
                path = os.path.join(sub, name)
                table = _load_table(run_dir, r["id"])
                audio, sr = _load_audio(run_dir, r.get("card_line", ""))
                info = renderer.render_clip(robot, table, profile, path, title="", subtitle="", audio=audio, sr=sr,
                                            card_seconds=0.0)
                entries.append({"rank": rank + 1, "id": r["id"], "blind_number": r["number"], "line_set": _set(r),
                                "sealed": is_sealed, "seed": r.get("seed"), "scores": r["scores"], "overall": r["overall"],
                                "description": redact_lines(r.get("description", ""), sealed) if is_sealed else r.get("description", ""),
                                "reason": redact_lines(r.get("reason", ""), sealed) if is_sealed else r.get("reason", ""),
                                "utterance": None if is_sealed else r.get("card_line"),
                                "file": os.path.relpath(path, out_dir).replace("\\", "/"), "seconds": info["seconds"]})
                log(f"[showcase] {mv}: {r['id']} overall {r['overall']:g} -> {os.path.relpath(path, out_dir)}")
            runners = [{"id": x["id"], "overall": x["overall"], "line_set": _set(x)} for x in ranked if x not in chosen][:3]
            manifest["movements"][mv] = {"exported": entries, "runners_up": runners,
                                         "vendor_reference": next(({"id": x["id"], "overall": x["overall"]} for x in records
                                                                   if x["robot"] == robot and x["source"] == VENDOR
                                                                   and x["movement"] == mv and x.get("overall") is not None), None)}
    with open(os.path.join(out_dir, "showcase.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest
