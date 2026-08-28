"""Perform the film's lines on the physical Reachy Mini, out loud.

The robot speaks a line (audio on the laptop's speaker) and moves with motion
that animacy generated from that same audio — so what you film is the product,
not a puppet show. Lines come from the film script (docs/video/script.md) or,
once the good voices exist, from the rendered narration manifest.

    python scripts/video/reachy_perform.py --host reachy --loop
    python scripts/video/reachy_perform.py --from-manifest data/video/voice/manifest.json --host reachy
    python scripts/video/reachy_perform.py --lines "Hey! Good to see you again." --repeat 3

Ctrl-C stops and parks the robot. Stop a detached run with `--stop`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from animacy.profile import find_robot  # noqa: E402
from animacy.retarget import retarget_clip  # noqa: E402
from animacy.serve import retrieval_motion  # noqa: E402
from animacy.sinks import make_sink, stream_table  # noqa: E402

STOP_FILE = os.path.join(ROOT, "data", "perform.stop")
LINE_RE = re.compile(r"^\*\*(LAMP|REACHY):\*\*\s*(.+?)\s*$")


def lines_from_script(path: str, host: str):
    want = {"reachy": "REACHY", "lamp": "LAMP", "both": None}[host]
    out = []
    for raw in open(path, encoding="utf-8"):
        m = LINE_RE.match(raw.strip())
        if m and (want is None or m.group(1) == want):
            text = m.group(2).replace("—", "-").replace("’", "'")
            out.append({"host": m.group(1).lower(), "text": text, "wav": None})
    return out


def lines_from_manifest(path: str, host: str):
    data = json.load(open(path, encoding="utf-8"))
    rows = data["lines"] if isinstance(data, dict) and "lines" in data else data
    out = []
    for r in rows:
        if host != "both" and r.get("host", "").lower() != host:
            continue
        wav = r.get("wav")
        if wav and not os.path.isabs(wav):
            wav = os.path.join(ROOT, wav)
        out.append({"host": r.get("host", host), "text": r.get("text", ""), "wav": wav})
    return out


def load_wav16k(path: str):
    import soundfile as sf
    from scipy.signal import resample_poly

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != 16000:
        from math import gcd

        g = gcd(int(sr), 16000)
        mono = resample_poly(mono, 16000 // g, int(sr) // g).astype(np.float32)
    return mono.astype(np.float32), 16000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="reachy_mini")
    ap.add_argument("--url", default="http://192.168.1.60:8000")
    ap.add_argument("--host", default="reachy", choices=["reachy", "lamp", "both"],
                    help="which speaker's lines to perform")
    ap.add_argument("--script", default=os.path.join(ROOT, "docs", "video", "script.md"))
    ap.add_argument("--from-manifest", default=None, help="use rendered narration wavs instead of live TTS")
    ap.add_argument("--lines", nargs="*", default=None, help="perform these texts instead of the script")
    ap.add_argument("--checkpoint", default=os.path.join(ROOT, "checkpoints", "v2a"))
    ap.add_argument("--gap", type=float, default=1.1)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--stop", action="store_true")
    a = ap.parse_args()

    if a.stop:
        open(STOP_FILE, "w").write("stop")
        print("stop requested")
        return 0
    if os.path.exists(STOP_FILE):
        os.remove(STOP_FILE)

    if a.lines:
        items = [{"host": a.host, "text": t, "wav": None} for t in a.lines]
    elif a.from_manifest:
        items = lines_from_manifest(a.from_manifest, a.host)
    else:
        items = lines_from_script(a.script, a.host)
    if not items:
        print("no lines found")
        return 1

    prof = find_robot(a.robot)
    sink = make_sink(prof, None, a.url)
    sink.prepare()
    print(f"[perform] {prof.name} <- {len(items)} line(s), audio={'off' if a.no_audio else 'on'}", flush=True)

    from animacy.tts import play_async, synth

    n = 0
    try:
        while not os.path.exists(STOP_FILE):
            for it in items:
                for _ in range(a.repeat):
                    if os.path.exists(STOP_FILE):
                        break
                    if it["wav"] and os.path.exists(it["wav"]):
                        wav, sr = load_wav16k(it["wav"])
                    else:
                        wav, sr = synth(it["text"])
                    clip = retrieval_motion(wav, sr, checkpoint=a.checkpoint, intent=it["text"], seed=n)
                    table = retarget_clip(clip, prof)
                    n += 1
                    print(f"[perform] {n:03d} {it['host']}: {it['text'][:64]}", flush=True)
                    player = None
                    if not a.no_audio:
                        try:
                            player = play_async(wav)
                        except Exception as e:  # noqa: BLE001
                            print("  (audio unavailable:", e, ")")
                    stream_table(table, prof, sink)
                    if player is not None:
                        try:
                            player.wait()
                        except Exception:  # noqa: BLE001
                            pass
                    time.sleep(a.gap)
            if not a.loop:
                break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            sink.neutral(1.0)
        except Exception:  # noqa: BLE001
            pass
        print(f"[perform] done after {n} line(s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
