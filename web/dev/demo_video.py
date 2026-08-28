"""60-second demo video of the viewer, frame-accurate (Playwright + ffmpeg).

    python web/dev/demo_video.py                 # docs/media/animacy_lamp_60s.mp4 + lamp_nod_ab_12s.mp4 + lamp_nod_ab.gif
    python web/dev/demo_video.py --headed --gpu  # Kokoro on WebGPU (faster TTS); same output
    python web/dev/demo_video.py --no-tts        # synthetic placeholder voice (the captions say so)

Nothing is screen-recorded. The page is parked (``animacy.setCapture(true)``)
and advanced one frame at a time (``animacy.stepFrame(1/30)``); every frame is
a screenshot, so frame i is exactly i/30 s of clip time however slow the
renderer is. Talk mode runs the real Kokoro TTS in the page on a manual clock;
the waveform is pulled out of the page and muxed at the frame the line started,
so voice and motion share one clock by construction (the same way the Python
runtime and the grader's renderer work). Captions are ffmpeg drawtext.

Shot list (docs/SUBMISSION.md):
  0-8 s   title over both robots idling (lamp hero; the lamp plays its vendor `idle`)
  8-20 s  vendor nod A/B on the lamp (human nod through ROBOT.md | vendor nod raw | same nod on Reachy)
  20-32 s calibration clips on both robots: look left/right, then brows
  32-50 s Talk: the excitement line, then the thinking line (retrieval, af_heart), both robots
  50-60 s end card
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from screenshot import free_port, wait_port  # noqa: E402

FPS = 30
W, H = 1280, 720
BG = "0x0e1117"
FONT = r"C:\Windows\Fonts\segoeui.ttf"
FONT_BOLD = r"C:\Windows\Fonts\segoeuib.ttf"
LINE_EXCITED = "No way, that is incredible news!"
LINE_THINKING = "Hmm... let me think about that for a moment."

# the suite's synthetic voice (24 kHz, 2.5 s): the fallback when TTS is unavailable
SYNTH_VOICE_JS = """(() => { const sr = 24000, n = Math.round(2.5 * sr), out = new Float32Array(n);
  for (let i = 0; i < n; i++) { const t = i / sr; const env = (t % 0.45) < 0.25 ? Math.sin(Math.PI * ((t % 0.45) / 0.25)) : 0;
    let v = 0; for (let k = 1; k < 6; k++) v += Math.sin(2 * Math.PI * 130 * k * t) / k; out[i] = 0.5 * env * v + 0.01 * (Math.random() - 0.5); }
  return Array.from(out); })()"""


def log(msg: str) -> None:
    print(msg, flush=True)


class Capture:
    """Frame-by-frame capture of the parked page."""

    def __init__(self, page, frames_dir: str):
        self.page = page
        self.dir = frames_dir
        self.n = 0
        self.t_shot = 0.0
        os.makedirs(frames_dir, exist_ok=True)

    def grab(self):
        t0 = time.perf_counter()
        self.page.screenshot(path=os.path.join(self.dir, f"{self.n:05d}.jpg"), type="jpeg", quality=93)
        self.t_shot += time.perf_counter() - t0
        self.n += 1

    def step(self, frames: int, dt: float = 1.0 / FPS):
        for _ in range(frames):
            self.page.evaluate(f"window.animacy.stepFrame({dt})")
            self.grab()

    def step_until(self, js_condition: str, max_frames: int, dt: float = 1.0 / FPS) -> int:
        k = 0
        while k < max_frames:
            self.page.evaluate(f"window.animacy.stepFrame({dt})")
            self.grab()
            k += 1
            if self.page.evaluate(js_condition):
                break
        return k

    @property
    def seconds(self) -> float:
        return self.n / FPS


def write_wav(path: str, sr: int, total_s: float, parts):
    """parts: [(offset_s, np.ndarray float32 at `sr`)] → 16-bit mono wav of `total_s`."""
    buf = np.zeros(int(round(total_s * sr)) + sr, np.float32)
    for off, a in parts:
        s = int(round(off * sr))
        e = min(len(buf), s + len(a))
        if e > s:
            buf[s:e] += a[: e - s]
    peak = float(np.max(np.abs(buf))) if len(buf) else 0.0
    if peak > 0.98:
        buf *= 0.98 / peak
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(buf, -1, 1) * 32767).astype("<i2").tobytes())


def resample(a: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return a.astype(np.float32)
    n_out = int(round(len(a) * sr_out / sr_in))
    x = np.linspace(0, len(a) - 1, n_out)
    return np.interp(x, np.arange(len(a)), a).astype(np.float32)


def drawtext(text_file: str, *, size: int, y: str, bold: bool = False, enable: str | None = None, box: bool = True,
             color: str = "white", x: str = "(w-text_w)/2", spacing: int = 6) -> str:
    font = "segoeuib.ttf" if bold else "segoeui.ttf"   # copied into the work dir: no drive-letter colon to escape
    s = (f"drawtext=fontfile={font}:textfile={text_file}:fontsize={size}:fontcolor={color}:x={x}:y={y}:line_spacing={spacing}"
         + (f":box=1:boxcolor=black@0.55:boxborderw=16" if box else ""))
    if enable:
        s += f":enable='{enable}'"
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "media", "animacy_lamp_60s.mp4"))
    ap.add_argument("--ab-out", default=os.path.join(ROOT, "docs", "media", "lamp_nod_ab_12s.mp4"))
    ap.add_argument("--gif", default=os.path.join(ROOT, "docs", "media", "lamp_nod_ab.gif"))
    ap.add_argument("--work", default=os.path.join(HERE, ".demo_video"), help="frames + intermediates (deleted unless --keep)")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--gpu", action="store_true", help="real GPU (d3d11) instead of SwiftShader; Kokoro can use WebGPU")
    ap.add_argument("--no-tts", action="store_true", help="synthetic placeholder voice instead of Kokoro")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--profile", default=os.path.join(HERE, ".chromium-profile"), help="persistent profile (caches the Kokoro download)")
    ap.add_argument("--skip-ab-clip", action="store_true")
    ap.add_argument("--crf", type=int, default=21)
    a = ap.parse_args()

    from playwright.sync_api import sync_playwright

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not on PATH")
    work = os.path.abspath(a.work)
    if os.path.isdir(work):
        shutil.rmtree(work)
    os.makedirs(work)
    for f in (FONT, FONT_BOLD):
        shutil.copy(f, os.path.join(work, os.path.basename(f)))

    port = free_port()
    srv = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    console_errors: list[str] = []
    benign = ("INFO: Created TensorFlow Lite", "[W:onnxruntime", "Unable to determine content-length")
    caveats: list[str] = []
    t_start = time.time()
    try:
        wait_port(port)
        with sync_playwright() as p:
            args = ["--autoplay-policy=no-user-gesture-required", "--ignore-gpu-blocklist"]
            if a.gpu:
                args += ["--use-angle=d3d11", "--enable-gpu-rasterization", "--enable-unsafe-webgpu"]
            else:
                args += ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
            ctx = p.chromium.launch_persistent_context(a.profile, headless=not a.headed, viewport={"width": W, "height": H},
                                                       device_scale_factor=1, args=args)
            page = ctx.new_page()
            page.set_default_timeout(900_000)
            page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" and not any(b in m.text for b in benign) else None)
            page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"))

            def ev(js: str):
                return page.evaluate(js)

            log(f"open the viewer on :{port} ({'headed' if a.headed else 'headless'}, {'gpu' if a.gpu else 'swiftshader'})")
            page.goto(f"http://127.0.0.1:{port}/web/?autoplay=0&source=native&clip=lamp/idle", wait_until="domcontentloaded")
            page.wait_for_function("window.animacy && window.animacy.ready === true", timeout=300_000)
            page.wait_for_timeout(600)
            lay = ev("window.animacy.layoutInfo()")
            log(f"layout {lay['layout']}: widths {json.dumps({k: round(v) for k, v in lay['widths'].items()})}, hero {lay['hero']}")
            if lay["layout"] != "hero":
                caveats.append("page did not open in the hero layout")
            ev("window.animacy.setCapture(true)")

            # ---------------- main reel ----------------
            cap = Capture(page, os.path.join(work, "frames"))
            audio_parts: list[tuple[float, np.ndarray]] = []
            sr_out = 24000
            marks: dict[str, float] = {}

            # S1 0-8 s: title over both robots idling; the lamp plays the vendor's `idle`
            if ev("window.animacy.sourceInfo().clip") != "lamp/idle":
                ev("(async () => { await window.animacy.setSource('native'); await window.animacy.setClip('lamp/idle'); })()")
                page.wait_for_function("window.animacy.sourceInfo().clip === 'lamp/idle'")
            ev("window.animacy.seek(0); window.animacy.play()")
            marks["s1"] = cap.seconds
            cap.step(8 * FPS)
            log(f"S1 title/idle done: {cap.n} frames, {cap.t_shot / max(cap.n, 1) * 1000:.0f} ms/shot")

            # S2 8-20 s: A/B vendor nod on the lamp
            ev("window.animacy.demo('ab')")
            page.wait_for_function("window.animacy.ab.on && window.animacy.ab.source && window.animacy.sourceInfo().clip === 'synth/cal_nod'")
            ev("window.animacy.seek(0); window.animacy.ab.source.seek(0); window.animacy.play()")
            page.wait_for_timeout(300)   # ResizeObserver → the three columns take their sizes
            marks["s2"] = cap.seconds
            cap.step(12 * FPS)
            log(f"S2 A/B done: {cap.n} frames")

            # S3 20-32 s: calibration clips on both robots (look left/right, then brows)
            ev("(async () => { await window.animacy.setAb(false); await window.animacy.setClip('synth/cal_look_left_right'); window.animacy.seek(0); window.animacy.play(); })()")
            page.wait_for_function("!window.animacy.ab.on && window.animacy.sourceInfo().clip === 'synth/cal_look_left_right'")
            page.wait_for_timeout(300)
            marks["s3a"] = cap.seconds
            cap.step(6 * FPS)
            ev("(async () => { await window.animacy.setClip('synth/cal_brows'); window.animacy.seek(0); window.animacy.play(); })()")
            page.wait_for_function("window.animacy.sourceInfo().clip === 'synth/cal_brows'")
            marks["s3b"] = cap.seconds
            cap.step(6 * FPS)
            log(f"S3 calibration done: {cap.n} frames")

            # S4 32-50 s: Talk — the excitement line, then the thinking line
            marks["s4"] = cap.seconds
            ev("window.animacy.demo('talk', {say: false})")
            page.wait_for_function("window.animacy.sourceInfo().kind === 'talk' && window.animacy.talk")
            ev(f"document.getElementById('talk-voice').value = {json.dumps(a.voice)}")
            tts_used = None
            for i, line in enumerate((LINE_EXCITED, LINE_THINKING)):
                said = None
                if not a.no_tts:
                    t0 = time.time()
                    try:
                        said = ev(f"window.animacy.say({json.dumps(line)})")
                        tts_used = "kokoro" if said else tts_used
                        log(f"  say({line!r}) → {said and {k: said[k] for k in ('backend', 'seconds', 'ttsMs', 'motionMs', 'intent')}} in {time.time() - t0:.1f} s")
                    except Exception as e:  # noqa: BLE001
                        log(f"  TTS failed: {e}")
                        said = None
                if not said:
                    tts_used = "synthetic"
                    said = ev(f"window.animacy.sayAudio({SYNTH_VOICE_JS}, 24000, 'retrieval', {json.dumps(line)})")
                    log(f"  placeholder voice for {line!r} → {said and {k: said[k] for k in ('backend', 'seconds', 'intent')}}")
                if not said:
                    raise RuntimeError(f"talk failed for {line!r}: {ev('window.animacy.errors')}")
                # the frame captured now is clip time 0 of this line: mux its audio here
                la = ev("(() => { const a = window.animacy.talk.lastAudio; return a ? {sr: a.sr, b64: btoa(String.fromCharCode(...new Uint8Array(new Int16Array(Array.from(a.audio, v => Math.max(-32767, Math.min(32767, Math.round(v * 32767))))).buffer)))} : null; })()")
                samples = np.frombuffer(base64.b64decode(la["b64"]), dtype="<i2").astype(np.float32) / 32767.0
                marks[f"line{i + 1}"] = cap.seconds
                audio_parts.append((cap.seconds, resample(samples, la["sr"], sr_out)))
                cap.step(1, dt=0.0)
                k = cap.step_until("window.animacy.talk.finished", max_frames=15 * FPS)
                marks[f"line{i + 1}_end"] = cap.seconds
                log(f"  line {i + 1}: {k} frames of motion ({k / FPS:.2f} s; speech {said['seconds']:.2f} s)")
                cap.step(int(0.8 * FPS))   # the settle after speech
            # hold to 50 s (the robots have settled); if the lines ran long, the reel just runs longer
            target = 50 * FPS
            if cap.n < target:
                cap.step(target - cap.n)
            marks["end"] = cap.seconds
            log(f"S4 talk done: {cap.n} frames ({cap.seconds:.1f} s); TTS = {tts_used}")
            if tts_used != "kokoro":
                caveats.append("voice is the synthetic placeholder (Kokoro TTS was unavailable in this render)")

            # ---------------- the 12 s A/B loop (lamp only) ----------------
            ab_frames = None
            if not a.skip_ab_clip:
                ab_frames = os.path.join(work, "ab_frames")
                ev("document.getElementById('vp-reachy_mini').hidden = true")
                ev("window.animacy.demo('ab')")
                page.wait_for_function("window.animacy.ab.on && window.animacy.ab.source && window.animacy.sourceInfo().clip === 'synth/cal_nod'")
                ev("window.animacy.seek(0); window.animacy.ab.source.seek(0); window.animacy.play()")
                page.wait_for_timeout(400)
                cap_ab = Capture(page, ab_frames)
                cap_ab.step(12 * FPS)
                ev("document.getElementById('vp-reachy_mini').hidden = false; window.animacy.setAb(false)")
                log(f"A/B loop captured: {cap_ab.n} frames")
            ev("window.animacy.setCapture(false)")
            ctx.close()
    finally:
        srv.terminate()
    if console_errors:
        caveats.append(f"{len(console_errors)} console error(s) during capture: {console_errors[:3]}")

    # ---------------- captions + encode ----------------
    reel_s = cap.seconds
    card_s = max(8.0, 60.0 - reel_s)
    total_s = reel_s + card_s
    wav = os.path.join(work, "audio.wav")
    write_wav(wav, sr_out, total_s, audio_parts)

    def tf(name: str, text: str) -> str:
        path = os.path.join(work, f"{name}.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return f"{name}.txt"

    voice_note = "" if tts_used == "kokoro" else "  (placeholder voice: TTS was unavailable in this render)"
    s1, s2, s3a, s3b, s4 = marks["s1"], marks["s2"], marks["s3a"], marks["s3b"], marks["s4"]
    l1, l1e, l2, l2e = marks["line1"], marks["line1_end"], marks["line2"], marks["line2_end"]
    filters = [
        "[0:v]"
        + drawtext(tf("title", "animacy"), size=76, y="96", bold=True, enable=f"between(t,{s1},{s2})")
        + "," + drawtext(tf("title2", "human motion in, any robot out"), size=34, y="190", enable=f"between(t,{s1},{s2})")
        + "," + drawtext(tf("cap1", "One human motion space. One ROBOT.md per body. The Autonomous Lamp and the Reachy Mini, on their real URDFs."),
                         size=27, y="h-92", enable=f"between(t,{s1},{s2})")
        + "," + drawtext(tf("cap2", "A: a human nod retargeted through the lamp's ROBOT.md   |   B: the vendor's own nod CSV, raw   |   right: the same human nod on Reachy Mini"),
                         size=25, y="h-92", enable=f"between(t,{s2},{s3a})")
        + "," + drawtext(tf("cap3a", "A human calibration clip on both robots: look left, then right.\nThe signs live in each ROBOT.md, never in the data."),
                         size=27, y="h-118", enable=f"between(t,{s3a},{s3b})")
        + "," + drawtext(tf("cap3b", "Brow raise: the lamp's head tips, Reachy's antennas lift.\nOne canonical channel, two bodies."),
                         size=27, y="h-118", enable=f"between(t,{s3b},{s4})")
        + "," + drawtext(tf("cap4", "Talk mode: text -> Kokoro TTS in the page -> speech features -> motion retrieval -> both robots, in sync" + voice_note),
                         size=25, y="h-92", enable=f"between(t,{s4},{l1})")
        + "," + drawtext(tf("cap4a", f"\"{LINE_EXCITED}\"   intent: excitement\nText -> Kokoro TTS in the page -> speech features -> motion retrieval -> both robots, in sync" + voice_note),
                         size=27, y="h-118", enable=f"between(t,{l1},{l2})")
        + "," + drawtext(tf("cap4b", f"\"{LINE_THINKING}\"   intent: thinking\nSame pipeline, quieter gesture tier, the settle after the sentence ends" + voice_note),
                         size=27, y="h-118", enable=f"between(t,{l2},{reel_s})")
        + "[main]",
        f"color=c={BG}:s={W}x{H}:r={FPS}:d={card_s:.3f}"
        + "," + drawtext(tf("end1", "one motion space  ·  one ROBOT.md per robot"), size=46, y="(h/2)-120", bold=True, box=False)
        + "," + drawtext(tf("end2", "hcoder10.github.io/animacy/web"), size=40, y="(h/2)-30", box=False, color="0xffb86b")
        + "," + drawtext(tf("end3", "github.com/Hcoder10/animacy  ·  Apache-2.0"), size=28, y="(h/2)+50", box=False, color="0xcfd5e1")
        + "," + drawtext(tf("end4", "Reachy Mini verified on a physical unit. No Lamp on hand: the Lamp path is checked against Autonomous OS's own route code."),
                         size=20, y="h-70", box=False, color="0x8a93a6")
        + "[card]",
        "[main][card]concat=n=2:v=1:a=0[v]",
    ]
    script = os.path.join(work, "filters.txt")
    with open(script, "w", encoding="utf-8") as fh:
        fh.write(";\n".join(filters) + "\n")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", "frames/%05d.jpg", "-i", "audio.wav",
           "-filter_complex_script", "filters.txt", "-map", "[v]", "-map", "1:a", "-t", f"{total_s:.3f}",
           "-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-r", str(FPS),
           "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", os.path.abspath(a.out)]
    log("encode: " + " ".join(cmd[-6:]))
    subprocess.run(cmd, check=True, cwd=work)

    if ab_frames:
        os.makedirs(os.path.dirname(os.path.abspath(a.ab_out)), exist_ok=True)
        cap_txt = tf("ab_cap", "A: a human nod through the lamp's ROBOT.md        B: the vendor's own nod, raw")
        vf = drawtext(cap_txt, size=26, y="h-84")
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", "ab_frames/%05d.jpg", "-vf", vf,
                        "-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-r", str(FPS),
                        "-movflags", "+faststart", os.path.abspath(a.ab_out)], check=True, cwd=work)
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", os.path.abspath(a.ab_out),
                        "-vf", "fps=15,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4",
                        "-loop", "0", os.path.abspath(a.gif)], check=True, cwd=work)

    def probe(path: str) -> str:
        out = subprocess.run([ffmpeg.replace("ffmpeg", "ffprobe"), "-v", "error", "-show_entries", "format=duration:stream=width,height,codec_name",
                              "-of", "json", path], capture_output=True, text=True)
        try:
            j = json.loads(out.stdout)
            st = j.get("streams", [])
            return f"{float(j['format']['duration']):.2f} s, {os.path.getsize(path) / 1e6:.1f} MB, " + ", ".join(f"{s.get('codec_name')}{' ' + str(s.get('width')) + 'x' + str(s.get('height')) if s.get('width') else ''}" for s in st)
        except Exception:  # noqa: BLE001
            return f"{os.path.getsize(path) / 1e6:.1f} MB"

    log(f"\nwrote {os.path.relpath(a.out, ROOT)}: {probe(a.out)}")
    if ab_frames:
        log(f"wrote {os.path.relpath(a.ab_out, ROOT)}: {probe(a.ab_out)}")
        log(f"wrote {os.path.relpath(a.gif, ROOT)}: {os.path.getsize(a.gif) / 1e6:.1f} MB")
    log(f"marks (s): {json.dumps({k: round(v, 2) for k, v in marks.items()})}; reel {reel_s:.2f} s + card {card_s:.2f} s; "
        f"capture {cap.t_shot:.0f} s of screenshots; total {time.time() - t_start:.0f} s")
    for c in caveats:
        log(f"CAVEAT: {c}")
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
