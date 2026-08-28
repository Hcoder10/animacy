"""Render the timeline.

Primary path is melt (MLT) rendering the same project file a human opens in
Kdenlive or Shotcut. If melt is missing or fails, an ffmpeg assembly of the
identical EDL takes over so a cut always comes out.

Also produces the derivatives (720p, the silent lamp loop) and the QC frames
we actually look at afterwards.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edit_common import (  # noqa: E402
    EDL, EDIT_DATA, MEDIA, SHOTS, QC, W, H, FPS,
    ff, run, probe, find_melt, BROLL_DIR,
)

MASTER = MEDIA / "animacy_demo.mp4"
MASTER_720 = MEDIA / "animacy_demo_720.mp4"
LAMP_LOOP = MEDIA / "animacy_lamp_loop.mp4"

MAX_MASTER_MB = 60.0
MAX_720_MB = 25.0

SCALE = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
         f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,"
         f"fps={FPS},setsar=1,format=yuv420p")


def mb(p: Path) -> float:
    return p.stat().st_size / 1e6 if p.exists() else 0.0


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------


def build_mix(edl: EDL, log: list[str], target_lufs: float = -16.0) -> Path | None:
    """Narration + room tone into one bed, levelled with a single static gain.

    A static gain rather than dynamic loudnorm: the delivery already has the
    shape we want, and a compressor would flatten the deadpan.
    """
    cues = [c for c in edl.narration if c.src and Path(c.src).exists()]
    if not cues:
        log.append("mix: no narration audio - rendering silent")
        return None

    inputs: list[str] = []
    chains: list[str] = []
    labels: list[str] = []
    for i, c in enumerate(sorted(cues, key=lambda c: c.start)):
        inputs += ["-i", str(c.src)]
        ms = int(round(c.start * 1000))
        chains.append(f"[{i}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
                      f"adelay={ms}|{ms}[n{i}]")
        labels.append(f"[n{i}]")

    n = len(labels)
    tone_src = next((r.get("src") for r in edl.roomtone if r.get("src")), None)
    if tone_src and Path(tone_src).exists():
        tone_chunks = []
        for j, r in enumerate(edl.roomtone):
            idx = n + j
            inputs += ["-stream_loop", "-1", "-i", str(tone_src)]
            ms = int(round(r["start"] * 1000))
            chains.append(
                f"[{idx}:a]atrim=0:{r['dur']:.3f},asetpts=PTS-STARTPTS,"
                f"afade=t=in:st=0:d=0.25,afade=t=out:st={max(0.0, r['dur'] - 0.25):.3f}:d=0.25,"
                f"aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"adelay={ms}|{ms}[t{j}]"
            )
            tone_chunks.append(f"[t{j}]")
        labels += tone_chunks

    total = edl.total
    chains.append(
        "".join(labels) + f"amix=inputs={len(labels)}:duration=longest:normalize=0,"
        f"apad,atrim=0:{total:.3f},asetpts=PTS-STARTPTS,"
        f"alimiter=limit=0.94:level=disabled[out]"
    )

    raw = EDIT_DATA / "mix_raw.wav"
    ff([*inputs, "-filter_complex", ";".join(chains), "-map", "[out]",
        "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(raw)])

    proc = ff(["-i", str(raw), "-af",
               f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=json",
               "-f", "null", "-"], check=False, quiet=True)
    blob = proc.stderr or ""
    gain = 0.0
    measured = None
    try:
        j = json.loads(blob[blob.rindex("{"):blob.rindex("}") + 1])
        measured = float(j["input_i"])
        if math.isfinite(measured):
            gain = max(-12.0, min(12.0, target_lufs - measured))
    except Exception:
        pass

    mix = EDIT_DATA / "mix.wav"
    ff(["-i", str(raw), "-af",
        f"volume={gain:.2f}dB,alimiter=limit=0.94:level=disabled",
        "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(mix)])

    verify = ff(["-i", str(mix), "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
                 "-f", "null", "-"], check=False, quiet=True)
    final_i = None
    try:
        vb = verify.stderr or ""
        final_i = float(json.loads(vb[vb.rindex("{"):vb.rindex("}") + 1])["input_i"])
    except Exception:
        pass
    log.append(
        f"mix: {len(cues)} narration cue(s)"
        + (f", {len(edl.roomtone)} room-tone segment(s)" if tone_src else "")
        + (f"; {measured:.1f} -> {final_i:.1f} LUFS" if measured is not None and final_i is not None
           else "")
    )
    raw.unlink(missing_ok=True)
    return mix


# --------------------------------------------------------------------------
# melt
# --------------------------------------------------------------------------


def render_melt(project: Path, out: Path, log: list[str], *,
                crf: int = 20, bitrate: str | None = None) -> bool:
    melt = find_melt()
    if not melt:
        log.append("melt: not installed - falling back to ffmpeg")
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = EDIT_DATA / "melt_master.mp4"
    args = [
        melt, "-progress2", "-profile", "atsc_1080p_30", str(project),
        "-consumer", f"avformat:{tmp}",
        "vcodec=libx264", "pix_fmt=yuv420p", "preset=medium",
        "g=60", "acodec=aac", "ab=192k", "ar=48000", "channels=2",
        "movflags=+faststart", "threads=0",
    ]
    args.append(f"vb={bitrate}" if bitrate else f"crf={crf}")
    proc = run(args, check=False, quiet=True)
    ok = tmp.exists() and probe(tmp)["dur"] > 1.0
    if not ok:
        tail = (proc.stderr or proc.stdout or "")[-1500:]
        log.append(f"melt: render failed (rc={proc.returncode}) {tail.strip()[-500:]}")
        return False
    shutil.move(str(tmp), str(out))
    log.append(f"melt: rendered {out.name} ({probe(out)['dur']:.2f}s, {mb(out):.1f} MB)")
    return True


# --------------------------------------------------------------------------
# ffmpeg assembly
# --------------------------------------------------------------------------


def _render_shot(shot, dst: Path, extra_tail: float = 0.0) -> Path:
    dur = shot.dur + extra_tail
    if shot.kind in ("slug", "endcard"):
        ff(["-loop", "1", "-framerate", str(FPS), "-i", str(shot.src),
            "-t", f"{dur:.3f}", "-vf", SCALE, "-an",
            "-c:v", "libx264", "-crf", "16", "-preset", "veryfast", str(dst)])
    else:
        ff(["-ss", f"{shot.src_in:.3f}", "-i", str(shot.src), "-an",
            "-vf", f"{SCALE},tpad=stop_mode=clone:stop_duration=6",
            "-t", f"{dur:.3f}",
            "-c:v", "libx264", "-crf", "16", "-preset", "veryfast", str(dst)])
    return dst


def render_ffmpeg(edl: EDL, mix: Path | None, out: Path, log: list[str], *,
                  crf: int = 21, bitrate: str | None = None) -> bool:
    SHOTS.mkdir(parents=True, exist_ok=True)
    shots = sorted([s for s in edl.shots if s.src], key=lambda s: s.start)
    if not shots:
        log.append("ffmpeg: nothing to assemble")
        return False

    diss = {round(d["at"], 3): d.get("actual", d["dur"]) for d in edl.dissolves}
    diss = {k: v for k, v in diss.items() if v > 0.02}

    # split into runs; a run ends where the next shot dissolves in
    runs: list[list] = [[]]
    tails: list[float] = []
    for i, s in enumerate(shots):
        runs[-1].append(s)
        nxt = shots[i + 1] if i + 1 < len(shots) else None
        if nxt is not None and round(nxt.start, 3) in diss:
            tails.append(diss[round(nxt.start, 3)])
            runs.append([])
    tails.append(0.0)

    # render each shot; the last shot of a run carries the dissolve tail
    files: list[list[Path]] = []
    for ri, run_shots in enumerate(runs):
        row = []
        for si, s in enumerate(run_shots):
            dst = SHOTS / f"r{ri:02d}_s{si:03d}.mp4"
            tail = tails[ri] if si == len(run_shots) - 1 else 0.0
            _render_shot(s, dst, extra_tail=tail)
            row.append(dst)
        files.append(row)

    inputs: list[str] = []
    chains: list[str] = []
    run_labels: list[str] = []
    idx = 0
    run_lens: list[float] = []
    for ri, row in enumerate(files):
        segs = []
        for f in row:
            inputs += ["-i", str(f)]
            segs.append(f"[{idx}:v]")
            idx += 1
        label = f"[run{ri}]"
        if len(segs) == 1:
            chains.append(f"{segs[0]}null{label}")
        else:
            chains.append("".join(segs) + f"concat=n={len(segs)}:v=1:a=0{label}")
        run_labels.append(label)
        run_lens.append(sum(probe(f)["dur"] for f in row))

    cur = run_labels[0]
    cur_len = run_lens[0]
    for ri in range(1, len(run_labels)):
        d = tails[ri - 1]
        offset = max(0.0, cur_len - d)
        nxt = f"[x{ri}]"
        chains.append(f"{cur}{run_labels[ri]}xfade=transition=fade:"
                      f"duration={d:.3f}:offset={offset:.3f}{nxt}")
        cur_len = cur_len + run_lens[ri] - d
        cur = nxt

    # lower thirds on top, each with its 6-frame fade
    fade = 6 / FPS
    for ti, t in enumerate(sorted(edl.titles, key=lambda t: t.start)):
        if not t.png or not Path(t.png).exists():
            continue
        inputs += ["-loop", "1", "-framerate", str(FPS), "-t", f"{t.dur:.3f}", "-i", str(t.png)]
        lab = f"[lt{ti}]"
        chains.append(
            f"[{idx}:v]format=rgba,fade=t=in:st=0:d={fade:.3f}:alpha=1,"
            f"fade=t=out:st={max(0.0, t.dur - fade):.3f}:d={fade:.3f}:alpha=1,"
            f"setpts=PTS-STARTPTS+{t.start:.3f}/TB{lab}"
        )
        idx += 1
        nxt = f"[o{ti}]"
        chains.append(f"{cur}{lab}overlay=0:0:enable='between(t,{t.start:.3f},"
                      f"{t.start + t.dur:.3f})':eof_action=pass{nxt}")
        cur = nxt

    chains.append(f"{cur}format=yuv420p,trim=0:{edl.total:.3f},setpts=PTS-STARTPTS[vout]")

    vcodec = ["-c:v", "libx264", "-preset", "medium", "-g", "60", "-pix_fmt", "yuv420p"]
    vcodec += (["-b:v", bitrate, "-maxrate", bitrate, "-bufsize", "8M"] if bitrate
               else ["-crf", str(crf)])

    args = [*inputs]
    if mix and mix.exists():
        args += ["-i", str(mix)]
    args += ["-filter_complex", ";".join(chains), "-map", "[vout]"]
    if mix and mix.exists():
        args += ["-map", f"{idx}:a", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                 "-shortest"]
    args += [*vcodec, "-movflags", "+faststart", str(out)]

    out.parent.mkdir(parents=True, exist_ok=True)
    ff(args, quiet=False)
    ok = out.exists() and probe(out)["dur"] > 1.0
    if ok:
        log.append(f"ffmpeg: assembled {out.name} "
                   f"({probe(out)['dur']:.2f}s, {mb(out):.1f} MB, "
                   f"{len(shots)} shots, {len(diss)} dissolve(s))")
    return ok


# --------------------------------------------------------------------------
# deliverables
# --------------------------------------------------------------------------


def fit_size(render_fn, out: Path, limit_mb: float, log: list[str], label: str) -> None:
    """Re-encode at a computed bitrate if the quality-targeted pass overshoots."""
    if not out.exists():
        return
    size = mb(out)
    if size <= limit_mb:
        log.append(f"{label}: {size:.1f} MB (under the {limit_mb:.0f} MB ceiling)")
        return
    dur = probe(out)["dur"] or 1.0
    budget_bits = (limit_mb * 0.92) * 8e6
    audio_bits = 192_000 * dur
    vb = max(600_000, (budget_bits - audio_bits) / dur)
    log.append(f"{label}: {size:.1f} MB over ceiling, re-encoding at {vb/1e6:.2f} Mbps")
    render_fn(f"{int(vb)}")
    log.append(f"{label}: now {mb(out):.1f} MB")


def make_720(master: Path, out: Path, log: list[str]) -> None:
    if not master.exists():
        return
    ff(["-i", str(master), "-vf", "scale=1280:720:flags=lanczos,format=yuv420p",
        "-c:v", "libx264", "-crf", "24", "-preset", "medium", "-g", "60",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out)])
    if mb(out) > MAX_720_MB:
        dur = probe(out)["dur"] or 1.0
        vb = max(400_000, ((MAX_720_MB * 0.9) * 8e6 - 128_000 * dur) / dur)
        ff(["-i", str(master), "-vf", "scale=1280:720:flags=lanczos,format=yuv420p",
            "-c:v", "libx264", "-b:v", str(int(vb)), "-maxrate", str(int(vb * 1.3)),
            "-bufsize", "4M", "-preset", "medium", "-g", "60",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out)])
    log.append(f"720p: {out.name} {mb(out):.1f} MB, {probe(out)['dur']:.2f}s")


# The loop is meant to be the lamp actually moving - the retargeted nod, A/B.
# A terminal recording that merely has "retarget" in its filename is not that,
# and shipping one as the README header would be a lie about what the project
# does, so the match has to be positive and the disqualifiers are hard.
LOOP_WANT = ["lamp", "viewer", "nod", "ab", "lean", "gaze"]
LOOP_BLOCK = ["csv", "terminal", "report", "table", "check", "robotmd", "robot_md",
              "score", "readback", "read_back", "evidence", "datareport", "manifest",
              "mapping", "preview", "bars"]


def pick_loop_source(edl: EDL, log: list[str]) -> tuple[Path, float] | None:
    """The lamp doing the retargeted nod, or nothing."""
    if not BROLL_DIR.exists():
        log.append("loop: no b-roll directory - skipped")
        return None
    cands: list[tuple[float, Path]] = []
    for p in sorted(BROLL_DIR.glob("**/*.mp4")):
        hay = (p.stem + " " + p.parent.name).lower().replace("-", "_")
        if any(b in hay for b in LOOP_BLOCK):
            continue
        hits = sum(1 for k in LOOP_WANT if k in hay)
        if not hits:
            continue
        d = probe(p)["dur"]
        if d < 2.0:
            continue
        cands.append((hits * 2 + min(d, 30) / 30, p))
    if not cands:
        log.append("loop: SKIPPED - no clip of the lamp doing the retargeted nod. "
                   "Needs a viewer A/B clip from the b-roll agent "
                   "(>=14 s, steady framing).")
        return None
    cands.sort(key=lambda x: -x[0])
    src = cands[0][1]
    return src, probe(src)["dur"]


def make_lamp_loop(edl: EDL, out: Path, log: list[str], length: float = 12.0) -> None:
    picked = pick_loop_source(edl, log)
    if not picked:
        return
    src, dur = picked
    if dur <= 0.5:
        log.append(f"loop: {src.name} is too short")
        return
    # loop the source up to length, then cross-fade the seam so it cycles cleanly
    seam = 0.5
    vf = (f"scale=1280:720:force_original_aspect_ratio=decrease,"
          f"pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black,fps={FPS},setsar=1,format=yuv420p")
    body = EDIT_DATA / "_loop_body.mp4"
    ff(["-stream_loop", "-1", "-i", str(src), "-an", "-t", f"{length + seam:.3f}",
        "-vf", vf, "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", str(body)])
    head = EDIT_DATA / "_loop_head.mp4"
    ff(["-i", str(body), "-t", f"{length:.3f}", "-an", "-c", "copy", str(head)],
       check=False)
    if not head.exists() or probe(head)["dur"] < length - 0.2:
        ff(["-i", str(body), "-t", f"{length:.3f}", "-an",
            "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", str(head)])
    # scale=out_range=tv + an explicit format keeps this out of the deprecated
    # full-range yuvj420p that browsers render inconsistently
    ff(["-i", str(head), "-an",
        "-vf", f"fade=t=in:st=0:d=0.35,fade=t=out:st={length - 0.35:.3f}:d=0.35,"
               f"scale=out_range=tv,format=yuv420p",
        "-c:v", "libx264", "-crf", "23", "-preset", "medium", "-g", "60",
        "-pix_fmt", "yuv420p", "-color_range", "tv",
        "-movflags", "+faststart", str(out)])
    for j in (body, head):
        j.unlink(missing_ok=True)
    log.append(f"loop: {out.name} {probe(out)['dur']:.2f}s, {mb(out):.1f} MB, from {src.name}")


# --------------------------------------------------------------------------
# QC
# --------------------------------------------------------------------------


def qc_frames(master: Path, edl: EDL, log: list[str], n: int = 15) -> list[Path]:
    """Frames at moments that matter: every lower-third, the end card, each
    dissolve midpoint, then spread the rest evenly."""
    if not master.exists():
        return []
    if QC.exists():
        shutil.rmtree(QC, ignore_errors=True)
    QC.mkdir(parents=True, exist_ok=True)
    dur = probe(master)["dur"]

    want: list[tuple[float, str]] = []
    for t in edl.titles:
        want.append((t.start + 1.2, f"title_{t.text.replace(' ', '_')}"))
    for d in edl.dissolves:
        want.append((d["at"] + d.get("actual", d["dur"]) / 2, f"dissolve_s{d['section']}"))
    endcard = next((s for s in edl.shots if s.kind == "endcard"), None)
    if endcard:
        want.append((endcard.start + 1.4, "endcard"))
    want.append((0.6, "open"))
    # one frame per host angle, so framing gets checked on every camera rather
    # than on whichever ones an even spread happened to land on
    seen_cam: set[str] = set()
    for s in sorted(edl.shots, key=lambda s: s.start):
        if s.kind == "cam" and s.label not in seen_cam:
            seen_cam.add(s.label)
            want.append((s.start + min(1.0, s.dur / 2), f"cam{s.label}"))

    spare = max(0, n - len(want))
    for i in range(spare):
        want.append((dur * (i + 1) / (spare + 1), f"even_{i:02d}"))

    out: list[Path] = []
    for t, name in sorted(want):
        t = max(0.05, min(dur - 0.1, t))
        dst = QC / f"{t:07.2f}s_{name}.png"
        ff(["-ss", f"{t:.3f}", "-i", str(master), "-frames:v", "1",
            "-vf", "scale=960:-1", str(dst)], check=False)
        if dst.exists():
            out.append(dst)
    log.append(f"qc: extracted {len(out)} frames to {QC}")
    return out


def black_frame_check(master: Path, edl: EDL, log: list[str]) -> None:
    """Catch a clip that renders black or freezes - the two ways a generated
    cut silently dies.

    Placeholder slugs and the end card are stills on purpose, so their spans
    are excluded; otherwise they drown the signal we actually want.
    """
    if not master.exists():
        return
    dur = probe(master)["dur"]
    expected = [(s.start - 0.2, s.end + 0.2) for s in edl.shots
                if s.kind in ("slug", "endcard")]
    expected.append((dur - 1.4, dur + 1.0))     # the tail to black

    def explained(t: float) -> bool:
        return any(a <= t <= b for a, b in expected)

    def starts(vf: str, key: str) -> list[float]:
        p = ff(["-i", str(master), "-vf", vf, "-an", "-f", "null", "-"],
               check=False, quiet=True)
        out = []
        for line in (p.stderr or "").splitlines():
            if key in line:
                try:
                    out.append(round(float(line.split(key)[1].split()[0]), 2))
                except Exception:
                    pass
        return out

    def from_source(t: float, vf: str, key: str) -> bool:
        """Does the clip on screen at time t have the same defect itself?

        Terminal recordings hold still while their output is readable, and
        reels of judged clips have black between them. Those are properties of
        the footage, not of the cut, and reporting them as edit faults buries
        the ones that are.
        """
        s = next((s for s in edl.shots if s.start <= t < s.end and s.src), None)
        if s is None or not Path(s.src).exists():
            return False
        want = s.src_in + (t - s.start)
        p = ff(["-i", str(s.src), "-vf", vf, "-an", "-f", "null", "-"],
               check=False, quiet=True)
        for line in (p.stderr or "").splitlines():
            if key in line:
                try:
                    if abs(float(line.split(key)[1].split()[0]) - want) < 0.6:
                        return True
                except Exception:
                    pass
        return False

    for label, vf, key in (
        ("black stretches", "blackdetect=d=0.5:pic_th=0.98:pix_th=0.10", "black_start:"),
        ("frozen picture", "freezedetect=n=-58dB:d=1.4", "freeze_start:"),
    ):
        hits = [t for t in starts(vf, key) if not explained(t)]
        inherited = [t for t in hits if from_source(t, vf, key)]
        introduced = [t for t in hits if t not in inherited]
        log.append(
            f"qc: {label} - {len(introduced) or 'no'} introduced by the edit"
            + (f" at {introduced}" if introduced else "")
            + (f"; {len(inherited)} inherited from the source footage" if inherited else "")
        )
