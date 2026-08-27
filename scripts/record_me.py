"""Guided webcam session: the most valuable data for a desk robot is a person at a
desk talking to a camera — and *listening* to one.

    python scripts/record_me.py --out data/clips --subject me [--quick]

Runs a sequence of short prompts. Each prompt is spoken aloud (Windows SAPI),
then `animacy capture --source 0` records that segment into its own clip.
Listening segments post-set `speaking = 0` (the microphone hears the podcast,
not you, so the VAD flag would be wrong); the role is stored in meta.json.

Tips: sit ~60-90 cm from the camera, face fully in frame, room lit from the
front. Look at the camera as if it were the person. Do not perform — just talk.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

# (slug, role, seconds, spoken prompt)
PROMPTS = [
    ("neutral", "speaking", 8, "First, look at the camera with a relaxed face for a few seconds. This is your neutral pose."),
    ("explain_project", "speaking", 90, "Explain what you are building this week, like you would to a friend who is not technical."),
    ("funny_story", "speaking", 60, "Tell a funny story that actually happened to you."),
    ("excited", "speaking", 45, "Now tell me about something you are genuinely excited about. Let it show."),
    ("frustrated", "speaking", 45, "Now something that annoyed you recently. Complain a little."),
    ("sad", "speaking", 40, "Now something you find a bit sad, quietly."),
    ("questions", "speaking", 60, "Answer out loud: what is your favourite food, where would you travel tomorrow, and what would you tell yourself five years ago?"),
    ("yes_no", "speaking", 40, "Say yes and no in as many different ways as you can. Agree, disagree, hesitate, be certain."),
    ("listen_podcast_1", "listening", 120, "Listening time. Start a podcast or a video on your phone, and just listen and react naturally. Do not talk."),
    ("listen_podcast_2", "listening", 120, "Keep listening. React as you normally would, nod, frown, laugh, look away when you think."),
    ("listen_disagree", "listening", 60, "Find something you disagree with, and listen to it. Do not talk, just react."),
    ("greetings", "speaking", 40, "Greet me a few different ways, then say goodbye a few different ways."),
]

QUICK = {"neutral", "explain_project", "listen_podcast_1"}


def say(text: str) -> None:
    safe = text.replace("'", "''")
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "Add-Type -AssemblyName System.Speech; "
                    f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{safe}')"],
                   check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "clips"))
    ap.add_argument("--subject", default="me")
    ap.add_argument("--source", default="0")
    ap.add_argument("--quick", action="store_true", help="3 short segments to test the setup")
    ap.add_argument("--only", default="", help="comma-separated slugs")
    ap.add_argument("--no-preview", action="store_true")
    a = ap.parse_args()

    prompts = [p for p in PROMPTS if (not a.quick or p[0] in QUICK) and (not a.only or p[0] in a.only.split(","))]
    total = sum(p[2] for p in prompts)
    print(f"{len(prompts)} segments, {total / 60:.1f} min. Ctrl-C between segments to stop.")
    say("Recording session. Sit facing the camera. Each prompt is spoken before its segment starts.")
    done = []
    for i, (slug, role, secs, prompt) in enumerate(prompts, 1):
        out = os.path.join(a.out, f"{a.subject}_{slug}")
        print(f"\n[{i}/{len(prompts)}] {slug} ({role}, {secs}s) -> {out}")
        say(prompt)
        say("Starting in three, two, one.")
        cmd = [PY, "-m", "animacy.cli", "capture", "--source", a.source, "-o", out, "--duration", str(secs),
               "--neutral-seconds", "1.0" if slug == "neutral" else "0"]
        if not a.no_preview:
            cmd.append("--preview")
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print("capture failed; stopping.")
            return r.returncode
        if role == "listening":
            fix = ("from animacy.schema import HumanClip; c=HumanClip.load(r'%s'); c.frames['speaking']=0.0; "
                   "c.meta['role']='listening'; c.save(r'%s'); print('  speaking forced to 0 (listening)')" % (out, out))
            subprocess.run([PY, "-c", fix], cwd=ROOT)
        else:
            fix = ("from animacy.schema import HumanClip; c=HumanClip.load(r'%s'); c.meta['role']='speaking'; c.save(r'%s')" % (out, out))
            subprocess.run([PY, "-c", fix], cwd=ROOT)
        done.append((slug, time.time() - t0))
        say("Got it.")
    say("Session complete. Thank you.")
    print("\nrecorded:", ", ".join(f"{s} ({t:.0f}s)" for s, t in done))
    return 0


if __name__ == "__main__":
    sys.exit(main())
