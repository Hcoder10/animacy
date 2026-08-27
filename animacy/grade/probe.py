"""Can the judge actually watch an MP4? A tiny experiment, run before any grading.

A 3 s reel (1 s title card "Clip 7" + the lamp's own 2 s ``nod`` clip) is put
in an otherwise empty workspace and Kimi is asked to report the title-card
text, describe the motion and say how many frames it could see. The result is
saved with the run (``probe.json``). If the card text or the motion is wrong,
``run.py`` falls back to contact sheets + joint plots and the report says so.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict

from ..export import read_autonomous_os_csv
from ..profile import find_robot
from . import kimi
from .render import ROOT, ViewerRenderer

PROBE_TITLE = "Clip 7"
PROBE_SUBTITLE = "The robot expresses: a nod"
PROBE_CLIP = os.path.join(ROOT, "robots", "lamp", "clips", "native", "nod.csv")

PROBE_PROMPT = """You are checking a video file. Watch the entire video at {path} (about 3 seconds, with an audio track).

Answer ONLY with JSON, no markdown:
{{
  "can_view_video": true or false (false if you could not open or view the video at all),
  "title_card_text": "the exact text shown on the title card at the start, or '' if none",
  "what_moved": "one or two sentences: what object is shown and what it does after the title card",
  "direction": "the main direction of the movement (e.g. up-down, left-right, none)",
  "frames_seen": integer, how many distinct video frames or samples you were given,
  "has_audio": true or false,
  "notes": "anything about the fidelity of what you could see"
}}"""


def make_probe_reel(renderer: ViewerRenderer, out_mp4: str) -> Dict:
    lamp = find_robot("lamp")
    table = read_autonomous_os_csv(PROBE_CLIP)
    return renderer.render_clip("lamp", table, lamp, out_mp4, title=PROBE_TITLE, subtitle=PROBE_SUBTITLE)


def judge_probe(reel_mp4: str, workspace: Path, timeout: int = 600) -> Dict:
    """Copy the reel into an empty workspace, ask, and score the answer."""
    workspace = Path(workspace)
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    dst = workspace / "probe.mp4"
    shutil.copyfile(reel_mp4, dst)
    result: Dict = {"reel": str(dst), "workspace_listing": kimi.workspace_listing(workspace)}
    try:
        ans = kimi.ask_json(PROBE_PROMPT.format(path=dst.resolve()), workspace, timeout=timeout)
    except kimi.KimiError as e:
        result.update({"ok": False, "error": str(e), "video_seen": False})
        return result
    result["answer"] = {k: v for k, v in ans.items() if not k.startswith("_")}
    result["raw"] = ans.get("_raw", "")
    result["seconds"] = ans.get("_seconds")
    title = str(ans.get("title_card_text", ""))
    text = (str(ans.get("what_moved", "")) + " " + str(ans.get("direction", ""))).lower()
    card_ok = bool(re.search(r"clip\s*7", title, re.I))
    # the vendor 'nod' dips the head: up/down, bob, nod, dip, tilt, pitch all count
    motion_ok = bool(re.search(r"nod|up[- ]?(and|/)?[- ]?down|down[- ]?(and|/)?[- ]?up|bob|dip|bow|pitch|tilt", text))
    result.update({
        "ok": True,
        "card_ok": card_ok,
        "motion_ok": motion_ok,
        "video_seen": bool(ans.get("can_view_video", False)) and card_ok and motion_ok,
        "frames_seen": ans.get("frames_seen"),
    })
    return result


def run_probe(run_dir: str, renderer: ViewerRenderer, workspace_root: Path, force: bool = False,
              timeout: int = 600) -> Dict:
    """Render + judge once per run (cached in ``<run_dir>/probe.json``)."""
    os.makedirs(run_dir, exist_ok=True)
    cache = os.path.join(run_dir, "probe.json")
    if os.path.exists(cache) and not force:
        return json.load(open(cache, encoding="utf-8"))
    reel = os.path.join(run_dir, "probe_reel.mp4")
    render_info = make_probe_reel(renderer, reel)
    res = judge_probe(reel, Path(workspace_root) / "probe")
    res["render"] = render_info
    with open(cache, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    return res
