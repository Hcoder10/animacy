"""B-roll from the real web viewer (web/index.html), driven through window.animacy.

    python scripts/video/broll_viewer.py --shots channels lean_in ab talk

The page is parked (`animacy.setCapture(true)`) and advanced one frame at a
time (`animacy.stepFrame(1/30)`), so frame i is exactly i/30 s of clip time
however slow the renderer is. Every joint angle on screen comes from the real
retargeter reading the real ROBOT.md; nothing is keyframed for the camera.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from broll_common import (  # noqa: E402
    FPS, OUT_DIR, ROOT, Capture, H, Server, W, encode, log, register, workdir,
)

PROFILE = os.path.join(ROOT, "web", "dev", ".chromium-profile")
LINE_EXCITED = "No way, that is incredible news!"
# the suite's synthetic stand-in voice, used only if Kokoro is unavailable in the render
SYNTH_VOICE_JS = """(() => { const sr = 24000, n = Math.round(2.5 * sr), out = new Float32Array(n);
  for (let i = 0; i < n; i++) { const t = i / sr; const env = (t % 0.45) < 0.25 ? Math.sin(Math.PI * ((t % 0.45) / 0.25)) : 0;
    let v = 0; for (let k = 1; k < 6; k++) v += Math.sin(2 * Math.PI * 130 * k * t) / k; out[i] = 0.5 * env * v + 0.01 * (Math.random() - 0.5); }
  return Array.from(out); })()"""


class Viewer:
    """The viewer open on a local server, parked and ready to be stepped."""

    def __init__(self, p, query: str, *, gpu: bool = False, persistent: bool = False,
                 dsf: int = 1):
        self.srv = Server().__enter__()
        args = ["--autoplay-policy=no-user-gesture-required", "--ignore-gpu-blocklist"]
        args += (["--use-angle=d3d11", "--enable-gpu-rasterization", "--enable-unsafe-webgpu"] if gpu
                 else ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        if persistent:
            self.ctx = p.chromium.launch_persistent_context(
                PROFILE, headless=True, viewport={"width": W, "height": H},
                device_scale_factor=dsf, args=args)
            self.page = self.ctx.new_page()
            self.browser = None
        else:
            self.browser = p.chromium.launch(headless=True, args=args)
            self.ctx = self.browser.new_context(viewport={"width": W, "height": H},
                                                device_scale_factor=dsf)
            self.page = self.ctx.new_page()
        # Playwright's Chromium reports a WebGPU adapter it cannot open a device on
        # (dxil.dll), and kokoro-js's wasm retry inherits the dead session. Hiding
        # navigator.gpu takes the wasm path from the start — the same Kokoro model,
        # the path a viewer without WebGPU gets.
        self.ctx.add_init_script("Object.defineProperty(navigator, 'gpu', {get: () => undefined});")
        self.page.set_default_timeout(600_000)
        self.errors: list[str] = []
        benign = ("INFO: Created TensorFlow Lite", "[W:onnxruntime", "Unable to determine content-length")
        self.page.on("console", lambda m: self.errors.append(m.text)
                     if m.type == "error" and not any(b in m.text for b in benign) else None)
        self.page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
        self.page.goto(self.srv.url(f"web/?{query}"), wait_until="domcontentloaded")
        self.page.wait_for_function("window.animacy && window.animacy.ready === true", timeout=300_000)
        self.page.wait_for_timeout(700)
        lay = self.page.evaluate("window.animacy.layoutInfo()")
        log(f"  layout {lay['layout']}, viewport widths "
            f"{json.dumps({k: round(v) for k, v in lay['widths'].items()})}")
        self.page.evaluate("window.animacy.setCapture(true)")

    def ev(self, js: str):
        return self.page.evaluate(js)

    def box(self, selector: str, pad: int = 0) -> dict:
        b = self.page.locator(selector).bounding_box()
        if not b:
            raise RuntimeError(f"{selector} has no box (hidden?)")
        x, y = max(0, b["x"] - pad), max(0, b["y"] - pad)
        return {"x": x, "y": y,
                "width": min(W - x, b["width"] + 2 * pad),
                "height": min(H - y, b["height"] + 2 * pad)}

    def close(self) -> None:
        try:
            self.ev("window.animacy.setCapture(false)")
        except Exception:  # noqa: BLE001
            pass
        self.ctx.close()
        if self.browser:
            self.browser.close()
        self.srv.__exit__(None, None, None)


def finish(cap: Capture, viewer: Viewer, out_name: str, **meta) -> dict:
    if viewer.errors:
        log(f"  NOTE: {len(viewer.errors)} console error(s): {viewer.errors[:2]}")
    out_path = os.path.join(OUT_DIR, out_name)
    encode(cap.dir, out_path)
    entry = register(out_name, **meta)
    shutil.rmtree(os.path.dirname(cap.dir), ignore_errors=True)
    return entry


# ---------------------------------------------------------------------------
# 2 — the 28 canonical channels moving
# ---------------------------------------------------------------------------
def shot_channels(p, clip: str, start_s: float, seconds: float) -> dict:
    # 2x so the ~980x175 readout band is still crisp once it is scaled to 1920 wide
    v = Viewer(p, f"autoplay=0&source=canonical&clip={clip}", dsf=2)
    try:
        v.ev(f"(async () => {{ await window.animacy.setSource('canonical'); "
             f"await window.animacy.setClip({json.dumps(clip)}); }})()")
        v.page.wait_for_function(f"window.animacy.sourceInfo().clip === {json.dumps(clip)}")
        info = v.ev("window.animacy.sourceInfo()")
        log(f"  clip {info['clip']}: {info['duration']:.1f} s; seeking to {start_s:g} s")
        v.ev(f"window.animacy.seek({start_s}); window.animacy.play()")
        v.page.wait_for_timeout(300)
        box = v.box("#channels-panel")   # no padding: the joints panel starts right of it
        log(f"  channel panel {box['width']:.0f}x{box['height']:.0f} at "
            f"({box['x']:.0f},{box['y']:.0f})")
        cap = Capture(v.page, os.path.join(workdir("channels"), "frames"))
        cap.step(int(seconds * FPS), js="window.animacy.stepFrame($DT)", clip=box)
        n = len([c for c in (v.ev("window.animacy.getChannels()") or {})])
        log(f"  {n} canonical channels live at the last frame")
        return finish(cap, v, "s2_channel_bars.mp4", section="2",
                      shows=f"The viewer's canonical readout while a captured human clip plays: all "
                            f"28 channels ({n} live) moving at 30 Hz — head 6-DoF, gaze, brows, eyes, "
                            f"mouth, smile, torso, arm, and the speaking flag. No robot in it.",
                      source=f"web/ Canonical clip tab, clip {clip}, from t={start_s:g} s "
                             f"(#channels-panel region)",
                      notes="The readout is a wide band (about 5.6:1), so the frame is the band "
                            "scaled to full width and centred on the film's background — crop or "
                            "overlay it as the edit prefers.")
    finally:
        v.close()


# ---------------------------------------------------------------------------
# 4 — retarget: the lean-in, and the vendor nod A/B
# ---------------------------------------------------------------------------
def shot_lean_in(p, seconds: float = 12.0) -> dict:
    clip = "synth/cal_lean_in"
    v = Viewer(p, f"autoplay=0&source=canonical&clip={clip}")
    try:
        v.ev(f"(async () => {{ await window.animacy.setSource('canonical'); "
             f"await window.animacy.setClip({json.dumps(clip)}); }})()")
        v.page.wait_for_function(f"window.animacy.sourceInfo().clip === {json.dumps(clip)}")
        v.ev("window.animacy.seek(0); window.animacy.play()")
        v.page.wait_for_timeout(300)
        box = v.box("#viewports")
        cap = Capture(v.page, os.path.join(workdir("lean_in"), "frames"))
        gaze = []
        for _ in range(int(seconds * FPS)):
            v.ev("window.animacy.stepFrame(%r)" % (1.0 / FPS))
            cap.grab(clip=box)
            g = v.ev("window.animacy.linkForward('lamp', 'head')")
            if g:
                gaze.append(g)
        if gaze:
            # the beam's world direction: how far the lamp's gaze wandered while it leant in
            dz = max(g["z"] for g in gaze) - min(g["z"] for g in gaze)
            dy = max(g["y"] for g in gaze) - min(g["y"] for g in gaze)
            log(f"  lamp head forward axis over the move: dz {dz:.3f}, dy {dy:.3f}")
        return finish(cap, v, "s4_lean_in.mp4", section="4",
                      shows="The `lean in` calibration clip through both ROBOT.md files: the lamp "
                            "translates toward the viewer while forward kinematics keeps its beam "
                            "pointed at the person, and the springs give the settle.",
                      source=f"web/ Canonical clip tab, clip {clip} (viewports region)")
    finally:
        v.close()


def loop_frames(duration_s: float, target_s: float) -> tuple[int, int]:
    """Whole clip periods covering `target_s`, so the result loops seamlessly.

    Both sources are looping the same clip, so capturing an exact whole number
    of periods from t=0 puts the last frame one step before the first — no cut
    to hide at the loop point.
    """
    period = max(1, int(round(duration_s * FPS)))
    reps = max(1, -(-int(round(target_s * FPS)) // period))
    return period * reps, reps


def shot_ab(p, seconds: float = 16.0, hero: bool = False) -> dict:
    """The vendor nod beside the retargeted human nod.

    `hero=True` drops the Reachy viewport for the lamp-only pair the README
    header loop wants. Framing is fixed for the whole take: the viewer's camera
    never moves and there are no cuts inside the clip, so it loops cleanly.
    """
    v = Viewer(p, "autoplay=0&source=native&clip=lamp/idle")
    try:
        if hero:
            v.ev("document.getElementById('vp-reachy_mini').hidden = true")
        v.ev("window.animacy.demo('ab')")
        v.page.wait_for_function("window.animacy.ab.on && window.animacy.ab.source "
                                 "&& window.animacy.sourceInfo().clip === 'synth/cal_nod'")
        v.ev("window.animacy.seek(0); window.animacy.ab.source.seek(0); window.animacy.play()")
        v.page.wait_for_timeout(600)          # ResizeObserver settles the A/B columns
        info = v.ev("window.animacy.sourceInfo()")
        n, reps = loop_frames(float(info["duration"]), seconds)
        box = v.box("#viewports")
        log(f"  A/B viewports {box['width']:.0f}x{box['height']:.0f}; clip {info['clip']} "
            f"{info['duration']:.2f} s -> {reps} whole loops, {n} frames ({n / FPS:.2f} s)")
        cap = Capture(v.page, os.path.join(workdir("ab_hero" if hero else "ab"), "frames"))
        cap.step(n, js="window.animacy.stepFrame($DT)", clip=box)
        name = "s4_ab_lamp_hero_loop.mp4" if hero else "s4_ab_vendor_nod.mp4"
        extra = "The lamp alone: A the retargeted human nod, B the vendor's own." if hero else \
            "Plus the same human nod on the Reachy Mini in the third panel."
        return finish(cap, v, name, section="4",
                      shows=f"A/B on one screen: a human nod retargeted through the lamp's "
                            f"ROBOT.md beside the vendor's own hand-authored `nod` clip played "
                            f"raw. {extra}",
                      source=f"web/ A/B mode (animacy.demo('ab')): synth/cal_nod vs the lamp's "
                             f"native nod.csv (viewports region)",
                      notes=f"{reps} whole loops of the {info['duration']:.2f} s clip captured from "
                            f"t=0, so it loops seamlessly. Fixed framing, no camera move, no cuts "
                            f"inside the take."
                            + (" Intended as the source for the README header loop." if hero else ""))
    finally:
        v.close()


# ---------------------------------------------------------------------------
# 5 — the interaction layer
# ---------------------------------------------------------------------------
def shot_talk(p, line: str = LINE_EXCITED, voice: str = "af_heart") -> dict:
    v = Viewer(p, "autoplay=0&source=native&clip=lamp/idle", persistent=True)
    try:
        v.ev("window.animacy.demo('talk', {say: false})")
        v.page.wait_for_function("window.animacy.sourceInfo().kind === 'talk' && window.animacy.talk")
        v.ev(f"document.getElementById('talk-voice').value = {json.dumps(voice)}")
        cap = Capture(v.page, os.path.join(workdir("talk"), "frames"))

        # clear the field and type the line, a couple of frames per keystroke
        v.page.click("#talk-text")
        v.page.keyboard.press("Control+A")
        v.page.keyboard.press("Delete")
        cap.step(int(0.5 * FPS), js="window.animacy.stepFrame($DT)")
        for ch in line:
            v.page.keyboard.type(ch)
            cap.step(2, js="window.animacy.stepFrame($DT)")
        cap.step(int(0.5 * FPS), js="window.animacy.stepFrame($DT)")

        # walk the pointer to the button rather than teleporting to it
        b = v.page.locator("#talk-say").bounding_box()
        v.page.mouse.move(b["x"] - 260, b["y"] + 120)
        for i in range(1, 13):
            v.page.mouse.move(b["x"] - 260 + (b["width"] / 2 + 260) * i / 12,
                              b["y"] + 120 - 120 * i / 12 + b["height"] / 2 * i / 12, steps=3)
            cap.step(1, js="window.animacy.stepFrame($DT)")
        v.page.mouse.down()
        cap.step(2, js="window.animacy.stepFrame($DT)")
        v.page.mouse.up()

        # the button press started the real pipeline; wait for it rather than firing a
        # second say(), which the page correctly rejects as "still working on the previous line"
        backend, said = "kokoro", None
        try:
            v.page.wait_for_function("window.animacy.talk && window.animacy.talk.last",
                                     timeout=420_000)
            said = v.ev("window.animacy.talkInfo().last")
        except Exception as e:  # noqa: BLE001
            log(f"  the Say it press produced no line in time: {str(e)[:200]}")
            log(f"  page errors: {v.ev('window.animacy.errors')}")
        if not said:
            backend = "synthetic"
            said = v.ev(f"window.animacy.sayAudio({SYNTH_VOICE_JS}, 24000, 'retrieval', "
                        f"{json.dumps(line)})")
        if not said:
            raise RuntimeError(f"talk failed: {v.ev('window.animacy.errors')}")
        log(f"  say({line!r}) -> voice {backend}, {said['seconds']:.2f} s, "
            f"intent {said.get('intent')}, motion backend {said.get('backend')}")
        # The talk source's own clock is the WebAudio node, which stays suspended in a
        # headless render, so `finished` never flips: step the motion the sentence's
        # own length instead, then hold for the settle.
        cap.step(1, js="window.animacy.stepFrame(0)")
        k = cap.step_until("window.animacy.stepFrame($DT)", "window.animacy.talk.finished",
                           max_frames=int(said["seconds"] * FPS) + 2)
        cap.step(int(1.6 * FPS), js="window.animacy.stepFrame($DT)")   # the settle after the sentence
        log(f"  {k} frames of motion ({k / FPS:.2f} s) for {said['seconds']:.2f} s of speech "
            f"({said['frames']} motion frames generated in {said['motionMs']:.0f} ms)")

        note = ("Kokoro-82M ran on wasm and took %.1f s of wall time to synthesise; the page is "
                "stepped a frame at a time, so that wait is not in the clip (clip time is exact, "
                "render time is not)." % (said.get("ttsMs", 0) / 1000.0)) if backend == "kokoro" else \
            "Voice is the suite's synthetic stand-in: Kokoro TTS was unavailable in this render."
        return finish(cap, v, "s5_talk.mp4", section="5",
                      shows=f"The Talk tab end to end: the line is typed, Say it is pressed, and both "
                            f"robots move on the same clock as the speech — text -> TTS in the page -> "
                            f"speech features -> {said.get('backend')} motion -> both ROBOT.md files. "
                            f"Intent read from the text as `{said.get('intent')}`.",
                      source=f'web/ Talk tab, "{line}", voice {voice}, backend {said.get("backend")}',
                      notes=note)
    finally:
        v.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", nargs="*", default=["channels", "lean_in", "ab", "ab_hero", "talk"],
                    choices=["channels", "lean_in", "ab", "ab_hero", "talk"])
    ap.add_argument("--ab-seconds", type=float, default=16.0,
                    help="minimum length of the A/B takes (rounded up to whole clip loops)")
    ap.add_argument("--channels-clip", default="clip/zachary_levi_about_working_on_broadway_at_nerdhq")
    ap.add_argument("--channels-start", type=float, default=28.0)
    ap.add_argument("--seconds", type=float, default=12.0)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    from playwright.sync_api import sync_playwright

    failed = []
    with sync_playwright() as p:
        for name in a.shots:
            log(f"\n=== {name} ===")
            try:
                if name == "channels":
                    shot_channels(p, a.channels_clip, a.channels_start, a.seconds)
                elif name == "lean_in":
                    shot_lean_in(p, a.seconds)
                elif name == "ab":
                    shot_ab(p, a.ab_seconds)
                elif name == "ab_hero":
                    shot_ab(p, a.ab_seconds, hero=True)
                elif name == "talk":
                    shot_talk(p)
            except Exception as e:  # noqa: BLE001
                log(f"  FAILED {name}: {type(e).__name__}: {e}")
                failed.append(name)
    if failed:
        log(f"\nfailed shots: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
