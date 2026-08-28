"""Generated media for the edit: lower-thirds, the end card, placeholder slugs,
room tone, and levelled narration.

Everything here writes into data/video/edit/gen/ and is safe to delete; it is
all reproduced from the script and the EDL.
"""

from __future__ import annotations

import json
import math
import re
import sys
import wave
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edit_common import (  # noqa: E402
    EDIT_DATA, GEN, W, H, EDL, Shot, ff, probe,
)

FONTS = EDIT_DATA / "fonts"

# Inter is the house sans if we can get it; Segoe UI is a real humanist sans and
# a perfectly respectable fallback. Never a generic "Arial centred" look.
_SANS_CANDIDATES = [
    FONTS / "Inter-Variable.ttf",
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("C:/Windows/Fonts/calibri.ttf"),
]
_SANS_BOLD_CANDIDATES = list(_SANS_CANDIDATES)
_MONO_CANDIDATES = [
    Path("C:/Windows/Fonts/consola.ttf"),
    Path.home() / "AppData/Local/Microsoft/Windows/Fonts/UbuntuMono[wght].ttf",
    Path("C:/Windows/Fonts/cour.ttf"),
]

INTER_VF = "https://github.com/google/fonts/raw/main/ofl/inter/Inter%5Bopsz,wght%5D.ttf"
INTER_VF_NAME = "Inter-Variable.ttf"


def fetch_inter(log: list[str]) -> None:
    """Best-effort. If the network says no, we use Segoe UI and move on."""
    FONTS.mkdir(parents=True, exist_ok=True)
    dst = FONTS / INTER_VF_NAME
    if dst.exists() and dst.stat().st_size > 100_000:
        return
    try:
        import urllib.request
        req = urllib.request.Request(INTER_VF, headers={"User-Agent": "animacy-edit"})
        with urllib.request.urlopen(req, timeout=40) as r:
            data = r.read()
        if len(data) > 100_000:
            dst.write_bytes(data)
            log.append(f"fonts: Inter (variable) downloaded, {len(data)//1024} KB")
        else:
            log.append("fonts: Inter download was too small to be a font; using Segoe UI")
    except Exception as exc:
        log.append(f"fonts: Inter unavailable ({type(exc).__name__}); using Segoe UI")


def _font(cands: list[Path], size: int, weight: str | None = None):
    for c in cands:
        if not c.exists():
            continue
        try:
            f = ImageFont.truetype(str(c), size)
        except Exception:
            continue
        if weight:
            try:
                f.set_variation_by_name(weight)
            except Exception:
                pass
        return f
    return ImageFont.load_default()


def sans(size: int):
    return _font(_SANS_CANDIDATES, size, "Regular")


def sans_bold(size: int):
    return _font(_SANS_BOLD_CANDIDATES, size, "SemiBold")


def mono(size: int):
    return _font(_MONO_CANDIDATES, size)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "x"


# --------------------------------------------------------------------------
# lower thirds
# --------------------------------------------------------------------------

LT_MARGIN_X = 96
LT_BASELINE_UP = 92     # distance from the bottom edge to the text baseline
LT_SIZE = 34


def make_lower_third(text: str) -> Path:
    """Small, static, bottom-left. A 2px rule, the word, a soft shadow so it
    survives a bright frame. Nothing moves; the 6-frame fade is done in the
    timeline, not baked in."""
    GEN.mkdir(parents=True, exist_ok=True)
    out = GEN / f"lt_{_slug(text)}.png"

    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f = sans(LT_SIZE)

    bbox = d.textbbox((0, 0), text, font=f)
    th = bbox[3] - bbox[1]
    tw = bbox[2] - bbox[0]
    baseline_y = H - LT_BASELINE_UP
    text_y = baseline_y - th - bbox[1]
    rule_x = LT_MARGIN_X
    text_x = LT_MARGIN_X + 20

    # a soft dark wash behind the text only - keeps white legible on a bright
    # set without reading as a graphic box
    wash = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    wd = ImageDraw.Draw(wash)
    wd.rounded_rectangle(
        [rule_x - 30, text_y - 22, text_x + tw + 34, text_y + th + 22],
        radius=8, fill=(8, 9, 11, 70),
    )
    wash = wash.filter(ImageFilter.GaussianBlur(24))
    img = Image.alpha_composite(img, wash)
    d = ImageDraw.Draw(img)

    d.rectangle([rule_x, text_y + 1, rule_x + 2, text_y + th], fill=(255, 255, 255, 210))
    d.text((text_x + 1, text_y + 1), text, font=f, fill=(0, 0, 0, 150))
    d.text((text_x, text_y), text, font=f, fill=(255, 255, 255, 240))

    img.save(out)
    return out


# --------------------------------------------------------------------------
# end card
# --------------------------------------------------------------------------

END_LINES = [
    "animacy",
    "github.com/Hcoder10/animacy",
    "hcoder10.github.io/animacy/web",
]


def make_end_card(credit: str | None = None) -> Path:
    GEN.mkdir(parents=True, exist_ok=True)
    out = GEN / "endcard.png"
    img = Image.new("RGB", (W, H), (10, 10, 12))
    d = ImageDraw.Draw(img)

    f_title = sans_bold(78)
    f_url = mono(31)

    x = 168
    block_h = 78 + 34 + 44 + 44
    y = (H - block_h) // 2 - 10

    d.text((x, y), END_LINES[0], font=f_title, fill=(244, 244, 246))
    y += 78 + 40
    d.rectangle([x, y, x + 118, y + 2], fill=(96, 100, 108))
    y += 34
    for url in END_LINES[1:]:
        d.text((x, y), url, font=f_url, fill=(168, 172, 180))
        y += 44

    if credit:
        d.text((x, H - 96), credit, font=mono(20), fill=(110, 113, 120))

    img.save(out)
    return out


# --------------------------------------------------------------------------
# placeholder slugs
# --------------------------------------------------------------------------


def make_slug(shot: Shot) -> Path:
    GEN.mkdir(parents=True, exist_ok=True)
    key = _slug(f"{shot.label}_{shot.section}_{shot.note}")
    out = GEN / f"slug_{key}.png"
    if out.exists():
        return out

    img = Image.new("RGB", (W, H), (18, 19, 22))
    d = ImageDraw.Draw(img)
    d.rectangle([60, 60, W - 60, H - 60], outline=(52, 54, 60), width=2)

    f_head = mono(30)
    f_body = mono(44)
    f_small = mono(24)

    d.text((104, 116), "PLACEHOLDER \u2014 FOOTAGE NOT DELIVERED", font=f_head,
           fill=(214, 132, 84))
    d.text((104, 210), shot.note or shot.label, font=f_body, fill=(226, 228, 232))
    d.text((104, 286), f"section {shot.section}   {shot.dur:.2f}s   "
                       f"@ {shot.start:.2f}s", font=f_small, fill=(132, 136, 144))
    img.save(out)
    return out


# --------------------------------------------------------------------------
# room tone
# --------------------------------------------------------------------------


def make_room_tone(duration: float, target_dbfs: float = -50.0) -> Path:
    """Plain generated pink noise, scaled to the stated RMS. Honest about what
    it is: no 'atmosphere' library, no music."""
    GEN.mkdir(parents=True, exist_ok=True)
    out = GEN / "roomtone.wav"
    dur = max(2.0, duration)
    if out.exists() and probe(out)["dur"] >= dur - 0.2:
        return out

    import numpy as np

    sr = 48000
    n = int(sr * dur)
    rng = np.random.default_rng(7)
    white = rng.standard_normal(n)
    # Paul Kellet's pink filter. scipy if we have it, otherwise the same
    # difference equation by hand.
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1.0, -2.494956002, 2.017265875, -0.522189400]
    try:
        from scipy.signal import lfilter
        pink = lfilter(b, a, white)
    except Exception:
        pink = np.zeros(n)
        x1 = x2 = x3 = y1 = y2 = y3 = 0.0
        for i in range(n):
            x0 = white[i]
            y0 = (b[0] * x0 + b[1] * x1 + b[2] * x2 + b[3] * x3
                  - a[1] * y1 - a[2] * y2 - a[3] * y3)
            pink[i] = y0
            x3, x2, x1 = x2, x1, x0
            y3, y2, y1 = y2, y1, y0

    rms = float(np.sqrt(np.mean(pink ** 2))) or 1.0
    pink *= (10 ** (target_dbfs / 20.0)) / rms
    pink = np.clip(pink, -1.0, 1.0)
    stereo = np.stack([pink, pink], axis=1)
    pcm = (stereo * 32767.0).astype("<i2")

    with wave.open(str(out), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return out


# --------------------------------------------------------------------------
# narration levelling
# --------------------------------------------------------------------------


def level_narration(edl: EDL, log: list[str], target_lufs: float = -16.0) -> dict[int, str]:
    """One gain for the whole performance, not per-line loudnorm.

    Measuring each line separately and slamming each to -16 would flatten the
    delivery; we measure the concatenation once, then apply that single gain to
    every line so the relative dynamics survive.
    """
    src = [(c.line_idx, c.src) for c in edl.narration if c.src and Path(c.src).exists()]
    if not src:
        log.append("audio: no narration WAVs to level")
        return {}

    outdir = EDIT_DATA / "voice_norm"
    outdir.mkdir(parents=True, exist_ok=True)

    concat = EDIT_DATA / "_voice_concat.txt"
    concat.write_text(
        "".join(f"file '{Path(p).as_posix()}'\n" for _, p in src), encoding="utf-8"
    )
    measured = EDIT_DATA / "_voice_concat.wav"
    ff(["-f", "concat", "-safe", "0", "-i", str(concat),
        "-ac", "2", "-ar", "48000", str(measured)])

    proc = ff(["-i", str(measured), "-af",
               "loudnorm=I=%.1f:TP=-1.5:LRA=11:print_format=json" % target_lufs,
               "-f", "null", "-"], check=False, quiet=True)
    blob = proc.stderr or ""
    gain_db = 0.0
    try:
        j = json.loads(blob[blob.rindex("{"):blob.rindex("}") + 1])
        measured_i = float(j["input_i"])
        if math.isfinite(measured_i):
            gain_db = target_lufs - measured_i
        log.append(f"audio: narration measured {measured_i:.1f} LUFS, "
                   f"applying {gain_db:+.1f} dB to every line")
    except Exception:
        log.append("audio: loudness measurement failed, passing narration through flat")

    gain_db = max(-24.0, min(24.0, gain_db))
    mapping: dict[int, str] = {}
    for idx, p in src:
        dst = outdir / f"line_{idx:03d}.wav"
        ff(["-i", p, "-ac", "2", "-ar", "48000",
            "-af", f"volume={gain_db:.2f}dB,alimiter=limit=0.891:level=disabled",
            str(dst)])
        mapping[idx] = str(dst)

    for junk in (concat, measured):
        junk.unlink(missing_ok=True)
    return mapping


# --------------------------------------------------------------------------


def build_all(edl: EDL, log: list[str]) -> dict:
    fetch_inter(log)
    titles = {}
    for t in edl.titles:
        p = make_lower_third(t.text)
        t.png = str(p)
        titles[t.text] = str(p)

    endcard = make_end_card()
    slugs = {}
    for s in edl.shots:
        if s.kind == "slug":
            p = make_slug(s)
            s.src = str(p)
            slugs[s.label] = str(p)
        elif s.kind == "endcard":
            s.src = str(endcard)

    # one bed, long enough for the longest continuous host run, re-used
    longest = max((r["dur"] for r in edl.roomtone), default=0.0)
    tone = make_room_tone(longest + 3.0) if edl.roomtone else None
    if tone:
        for r in edl.roomtone:
            r["src"] = str(tone)

    voice = level_narration(edl, log)
    for c in edl.narration:
        if c.line_idx in voice:
            c.src = voice[c.line_idx]

    log.append(f"assets: {len(titles)} lower-third(s), {len(slugs)} slug(s), "
               f"end card, room tone={'yes' if tone else 'no'}")
    return {"endcard": str(endcard), "roomtone": str(tone) if tone else None,
            "titles": titles, "slugs": slugs}


if __name__ == "__main__":
    from edit_common import build_edl, save_edl
    e = build_edl()
    info = build_all(e, e.log)
    save_edl(e)
    print(json.dumps(info, indent=2))
    print("\n".join(e.log))
