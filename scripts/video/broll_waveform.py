"""B-roll for section 5: the speech waveform and the motion it produced, one clock.

    python scripts/video/broll_waveform.py [--movement excitement]

Both traces come out of the same graded clip: the audio is the TTS the grader
actually synthesised (pulled from that clip's reel part), the joint curves are
the joint table `animacy` generated from it (data/grading/<run>/clips/*.json).
The playhead is the shared clock — the figure plots them, it does not align them.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from broll_common import FPS, H, OUT_DIR, ROOT, W, encode, ffmpeg_bin, log, register, workdir  # noqa: E402

sys.path.insert(0, ROOT)
from animacy.grade.reel import CARD_SECONDS  # noqa: E402

BG, FG, DIM, GRID = "#14161b", "#d6dae1", "#8b93a1", "#252932"
WAVE_C, TRACE_C = "#7aa2d8", "#d9a05b"
HEAD = "#e8ebef"


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        sr, n, ch, width = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    dtype = {1: np.uint8, 2: "<i2", 4: "<i4"}[width]
    a = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if width == 1:
        a = (a - 128.0) / 128.0
    else:
        a /= float(2 ** (8 * width - 1))
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="20260827_1501_run3")
    ap.add_argument("--robot", default="lamp")
    ap.add_argument("--movement", default="excitement")
    ap.add_argument("--source", default="retrieval")
    ap.add_argument("--lines", default="heldout")
    ap.add_argument("--out", default="s5_waveform_motion.mp4")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = os.path.join(ROOT, "data", "grading", a.run)
    at = "@heldout" if a.lines == "heldout" else ""
    name = f"{a.robot}__{a.movement}{at}__{a.source}__s0.json"
    with open(os.path.join(run_dir, "clips", name), encoding="utf-8") as fh:
        clip = json.load(fh)
    clip_id = clip["id"]

    # find this clip's number on the reels, so we can pull its own TTS audio
    with open(os.path.join(run_dir, "manifest_sealed.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    num = next(k for k, v in man["robots"][a.robot]["clips"].items() if v["id"] == clip_id)
    part = os.path.join(run_dir, "reels", "parts", f"{a.robot}_clip{int(num):03d}.mp4")
    if not os.path.exists(part):
        raise SystemExit(f"no reel part for clip {num} ({clip_id}) at {part}")
    work = workdir("waveform")
    wav = os.path.join(work, "speech.wav")
    # the reel part is CARD_SECONDS of title card, then the clip: drop the card so the
    # audio starts where the joint table starts, which is what makes the shared clock true
    subprocess.run([ffmpeg_bin(), "-y", "-loglevel", "error", "-ss", f"{CARD_SECONDS:.3f}",
                    "-i", part, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav],
                   check=True)
    audio, sr = read_wav(wav)
    audio = audio[:int(round(clip["duration"] * sr))]
    log(f"  {clip_id}: {clip['duration']:.2f} s of motion, {len(audio) / sr:.2f} s of speech "
        f"from {os.path.basename(part)} (card {CARD_SECONDS:g} s trimmed)")

    t = np.asarray(clip["t"], dtype=np.float64)
    # the two axes that carry a lamp nod (ROBOT.md: base_pitch + wrist_pitch)
    traces = [(k, np.asarray(clip["data"][k], dtype=np.float64))
              for k in ("base_pitch", "wrist_pitch", "base_yaw") if k in clip["data"]]
    ta = np.arange(len(audio)) / sr
    span = max(float(t[-1]), float(ta[-1]))
    lead, tail = 0.7, 1.3
    total = lead + span + tail
    n_frames = int(round(total * FPS))
    log(f"  figure: {total:.2f} s ({n_frames} frames), {len(traces)} joint traces")

    plt.rcParams.update({
        "font.family": ["Cascadia Mono", "Consolas", "DejaVu Sans Mono", "monospace"],
        "text.color": FG, "axes.labelcolor": FG, "xtick.color": DIM, "ytick.color": DIM,
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    })
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W / 100, H / 100), dpi=100,
                                   gridspec_kw={"height_ratios": [1.0, 1.25], "hspace": 0.22})
    fig.subplots_adjust(left=0.075, right=0.975, top=0.855, bottom=0.085)
    line = clip["card_line"].replace("The robot says: ", "")
    fig.text(0.075, 0.945, f"speech {line}", fontsize=25, color=HEAD, va="top")
    fig.text(0.075, 0.895,
             f"intent {clip['meta']['intent']['tag']}  ·  amplitude x{clip['meta']['amplitude']}  ·  "
             f"{a.source} motion at {clip['meta']['rate_hz']:.0f} Hz  ·  one clock, no alignment step",
             fontsize=19, color=DIM, va="top")

    ax1.plot(ta, audio, color=WAVE_C, lw=0.8, alpha=0.95)
    ax1.set_ylabel("speech", fontsize=20, labelpad=12)
    ax1.set_ylim(-1.05 * float(np.max(np.abs(audio)) or 1), 1.05 * float(np.max(np.abs(audio)) or 1))
    for k, v in traces:
        ax2.plot(t, v, lw=2.6, label=k, alpha=0.95)
    ax2.set_ylabel("lamp joints (deg)", fontsize=20, labelpad=12)
    ax2.set_xlabel("seconds", fontsize=19, labelpad=10)
    leg = ax2.legend(loc="upper right", fontsize=18, frameon=False, ncol=len(traces))
    for txt in leg.get_texts():
        txt.set_color(FG)
    for ax in (ax1, ax2):
        ax.set_xlim(-0.05, span + 0.05)
        ax.grid(True, color=GRID, lw=0.9)
        ax.tick_params(labelsize=17)
        for s in ax.spines.values():
            s.set_color(GRID)
    ax1.tick_params(labelbottom=False)
    heads = [ax.axvline(0.0, color=TRACE_C, lw=2.2, alpha=0.0) for ax in (ax1, ax2)]

    frames = os.path.join(work, "frames")
    os.makedirs(frames, exist_ok=True)
    for i in range(n_frames):
        now = i / FPS - lead
        for h in heads:
            h.set_xdata([now, now])
            h.set_alpha(0.0 if now < 0 or now > span else 0.95)
        fig.savefig(os.path.join(frames, f"{i:05d}.jpg"), format="jpg", pil_kwargs={"quality": 93})
    plt.close(fig)

    out_path = os.path.join(OUT_DIR, a.out)
    encode(frames, out_path)
    register(a.out, section="5",
             shows=f"The speech and the motion it produced, on one time axis: the TTS waveform for "
                   f"\"{line}\" above, and the lamp joint angles animacy generated from it below, "
                   f"with a shared playhead. Both come from the same graded clip, so the sync is "
                   f"how they were made, not something the plot arranged.",
             source=f"data/grading/{a.run}/clips/{name} (motion) + reels/parts/"
                    f"{os.path.basename(part)} (its own TTS audio)",
             extra={"clip_id": clip_id, "movement": a.movement, "motion_source": a.source})
    import shutil
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
