"""Write the edit out as an MLT XML project.

Two files come out of here, from one EDL:

  docs/video/edit/animacy_demo.kdenlive   the project a human opens and tweaks
  data/video/edit/animacy_demo.mlt        the same timeline, with whatever
                                          composite service the local melt
                                          actually has, for headless rendering

Track layout (bottom to top, as MLT indexes them):

  0  background   black, full length - everything composites down onto it
  1  A1           narration, the spine
  2  A2           room tone, only under the host scenes
  3  V1           the picture
  4  V2           incoming clips at a dissolve only; blank the rest of the time
  5  V3           lower-thirds

Transitions, in the order MLT applies them: audio mixes down to 0, then the
luma dissolves fold V2 into V1, then V1 composites onto the background, then
the lower-thirds go on top.
"""

from __future__ import annotations

import sys
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edit_common import (  # noqa: E402
    EDL, FPS, W, H, REPO, frames, probe, run,
)

PROFILE = dict(
    description="HD 1080p 30 fps", width=str(W), height=str(H),
    progressive="1", sample_aspect_num="1", sample_aspect_den="1",
    display_aspect_num="16", display_aspect_den="9",
    frame_rate_num=str(FPS), frame_rate_den="1", colorspace="709",
)


def _res(path: str | Path) -> str:
    return Path(path).resolve().as_posix()


class _Producers:
    """One producer per distinct source file; entries re-use it with in/out."""

    def __init__(self):
        self.by_key: dict[tuple[str, str], str] = {}
        self.elements: list[ET.Element] = []
        self._n = 0
        self._kid = 2

    def get(self, path: str, kind: str, total_frames: int) -> str:
        key = (_res(path), kind)
        if key in self.by_key:
            return self.by_key[key]
        pid = f"producer{self._n}"
        self._n += 1
        length = max(2, total_frames)
        p = ET.Element("producer", {"id": pid, "in": "0", "out": str(length - 1)})
        props = {"length": str(length), "eof": "pause", "resource": _res(path)}
        if kind == "image":
            props["mlt_service"] = "qimage"
            props["ttl"] = "1"
            props["aspect_ratio"] = "1"
        elif kind == "audio":
            props["mlt_service"] = "avformat"
            props["video_index"] = "-1"
        else:
            props["mlt_service"] = "avformat"
            props["audio_index"] = "-1"
        props["kdenlive:id"] = str(self._kid)
        props["kdenlive:clipname"] = Path(path).name
        self._kid += 1
        for k, v in props.items():
            ET.SubElement(p, "property", {"name": k}).text = v
        self.elements.append(p)
        self.by_key[key] = pid
        return pid

    def color(self, name: str, colour: str, total_frames: int) -> str:
        pid = name
        p = ET.Element("producer", {"id": pid, "in": "0", "out": str(total_frames - 1)})
        for k, v in {
            "length": str(total_frames), "eof": "pause", "resource": colour,
            "mlt_service": "color", "aspect_ratio": "1",
            "mlt_image_format": "rgba",
        }.items():
            ET.SubElement(p, "property", {"name": k}).text = v
        self.elements.append(p)
        return pid


def _playlist(pid: str, *, audio: bool = False, name: str = "") -> ET.Element:
    pl = ET.Element("playlist", {"id": pid})
    if audio:
        ET.SubElement(pl, "property", {"name": "kdenlive:audio_track"}).text = "1"
    if name:
        ET.SubElement(pl, "property", {"name": "kdenlive:track_name"}).text = name
    return pl


_overlaps: list[str] = []


def _fill(pl: ET.Element, cursor: int, want: int, what: str = "") -> int:
    """Pad a playlist out to `want`. If the cursor is already past it the clip
    cannot start where it should, and everything after it slides - so that is
    recorded rather than silently absorbed."""
    if want > cursor:
        ET.SubElement(pl, "blank", {"length": str(want - cursor)})
        return want
    if want < cursor and what:
        _overlaps.append(f"{what}: wanted frame {want}, cursor at {cursor}")
    return cursor


def _transition(tid: str, service: str, a: int, b: int, *, always: bool = True,
                in_f: int | None = None, out_f: int | None = None,
                extra: dict | None = None) -> ET.Element:
    attrs = {"id": tid}
    if in_f is not None:
        attrs["in"] = str(in_f)
        attrs["out"] = str(out_f)
    t = ET.Element("transition", attrs)
    props = {"a_track": str(a), "b_track": str(b), "mlt_service": service,
             "factory": "loader"}
    if always:
        props["always_active"] = "1"
    if service == "mix":
        props.update({"sum": "1"})
    if service in ("frei0r.cairoblend", "qtblend", "composite"):
        props.update({"disable": "0"})
        if service == "composite":
            props.update({"fill": "1", "distort": "0"})
    if service == "luma":
        props.update({"kdenlive_id": "dissolve"})
    if extra:
        props.update(extra)
    for k, v in props.items():
        ET.SubElement(t, "property", {"name": k}).text = str(v)
    return t


def pick_composite(melt: str | None) -> str:
    """What this melt can actually do. Kdenlive's default is cairoblend."""
    if not melt:
        return "frei0r.cairoblend"
    try:
        out = run([melt, "-query", "transitions"], check=False, quiet=True).stdout or ""
    except Exception:
        return "frei0r.cairoblend"
    for cand in ("frei0r.cairoblend", "qtblend", "composite"):
        if cand in out:
            return cand
    return "frei0r.cairoblend"


def write_project(edl: EDL, out_path: Path, *, composite: str = "frei0r.cairoblend",
                  title: str = "animacy demo", log: list[str] | None = None) -> Path:
    _overlaps.clear()
    total_f = max(2, frames(edl.total))
    prods = _Producers()

    v1 = _playlist("playlist_v1", name="V1")
    v2 = _playlist("playlist_v2", name="V2 (dissolve)")
    v3 = _playlist("playlist_v3", name="V3 (lower thirds)")
    a1 = _playlist("playlist_a1", audio=True, name="A1 narration")
    a2 = _playlist("playlist_a2", audio=True, name="A2 room tone")

    dissolve_at = {round(d["at"], 3): d["dur"] for d in edl.dissolves}

    # ---- V1 / V2 ---------------------------------------------------------
    #
    # A dissolve at cut T spans [T, T+d]. The incoming picture starts at T at
    # its natural in-point, so it stays in sync with the narration; the
    # outgoing clip is what gets stretched over the overlap and faded away.
    # Doing it the other way round would slide the incoming shot d seconds
    # ahead of its own audio for the whole shot, and these robots are lip-
    # synced to the narration.
    v1_cursor = 0
    v2_cursor = 0
    last_entry: ET.Element | None = None
    last_cap = 0
    for shot in sorted(edl.shots, key=lambda s: s.start):
        if not shot.src:
            continue
        kind = "image" if shot.kind in ("slug", "endcard") else "video"
        src_dur = probe(shot.src)["dur"] if kind == "video" else edl.total
        if kind == "image":
            src_frames = max(2, frames(edl.total))
        else:
            # a second of headroom so a dissolve can hold the last frame
            # (eof=pause) rather than running off the end of the clip
            src_frames = max(2, frames(src_dur if src_dur > 0 else shot.dur + 1)
                             + frames(1.0))
        pid = prods.get(shot.src, kind, src_frames)

        start_f = frames(shot.start)
        end_f = frames(shot.end)
        in_f = frames(shot.src_in)
        d = dissolve_at.get(round(shot.start, 3))
        ov = frames(d) if d else 0

        if ov and last_entry is not None:
            o = int(last_entry.get("out"))
            new_out = min(o + ov, last_cap)
            last_entry.set("out", str(new_out))
            v1_cursor += new_out - o
            ov = new_out - o
        elif ov:
            ov = 0

        if ov:
            v2_cursor = _fill(v2, v2_cursor, start_f)
            head_out = min(in_f + ov - 1, src_frames - 1)
            ET.SubElement(v2, "entry", {"producer": pid, "in": str(in_f),
                                        "out": str(head_out)})
            v2_cursor += head_out - in_f + 1
            body_start = start_f + ov
            body_in = min(in_f + ov, src_frames - 2)
        else:
            body_start, body_in = start_f, in_f

        for dd in edl.dissolves:
            if round(dd["at"], 3) == round(shot.start, 3):
                dd["actual"] = round(ov / FPS, 3)

        v1_cursor = _fill(v1, v1_cursor, body_start)
        want = max(1, end_f - body_start)
        out_f = min(body_in + want - 1, src_frames - 1)
        if out_f <= body_in:
            body_in, out_f = 0, min(want - 1, src_frames - 1)
        entry = ET.SubElement(v1, "entry",
                              {"producer": pid, "in": str(body_in), "out": str(out_f)})
        v1_cursor += out_f - body_in + 1
        last_entry, last_cap = entry, src_frames - 1

    # ---- V3 lower thirds -------------------------------------------------
    c3 = 0
    for t in sorted(edl.titles, key=lambda t: t.start):
        if not t.png:
            continue
        pid = prods.get(t.png, "image", total_f)
        s, e = frames(t.start), frames(t.start + t.dur)
        c3 = _fill(v3, c3, s)
        n = max(2, e - s)
        entry = ET.SubElement(v3, "entry", {"producer": pid, "in": "0", "out": str(n - 1)})
        fade = max(2, frames(6 / FPS))
        _filter(entry, "brightness", {
            "alpha": f"0=0;{fade}=1;{n - 1 - fade}=1;{n - 1}=0",
            "opacity": f"0=0;{fade}=1;{n - 1 - fade}=1;{n - 1}=0",
            "mlt_image_format": "rgba",
        })
        c3 += n

    # ---- A1 narration ----------------------------------------------------
    ca = 0
    for cue in sorted(edl.narration, key=lambda c: c.start):
        if not cue.src or not Path(cue.src).exists():
            ca = _fill(a1, ca, frames(cue.start + cue.dur))
            continue
        n = max(2, frames(probe(cue.src)["dur"] or cue.dur))
        pid = prods.get(cue.src, "audio", n)
        ca = _fill(a1, ca, frames(cue.start), f"narration line {cue.line_idx}")
        use = max(1, min(n, frames(cue.dur))) - 1
        ET.SubElement(a1, "entry", {"producer": pid, "in": "0", "out": str(max(1, use))})
        ca += max(2, use + 1)

    # ---- A2 room tone ----------------------------------------------------
    tone_src = next((r.get("src") for r in edl.roomtone if r.get("src")), None)
    cb = 0
    if tone_src and Path(tone_src).exists():
        tn = max(2, frames(probe(tone_src)["dur"]))
        pid = prods.get(tone_src, "audio", tn)
        for r in edl.roomtone:
            s, n = frames(r["start"]), max(2, frames(r["dur"]))
            cb = _fill(a2, cb, s)
            entry = ET.SubElement(a2, "entry",
                                  {"producer": pid, "in": "0", "out": str(min(n, tn) - 1)})
            fade = max(2, frames(0.25))
            _filter(entry, "volume", {"level": f"0=-60;{fade}=0;{n - 1 - fade}=0;{n - 1}=-60"})
            cb += min(n, tn)

    # ---- assemble --------------------------------------------------------
    mlt = ET.Element("mlt", {
        "LC_NUMERIC": "C", "version": "7.30.0", "title": title,
        "root": REPO.resolve().as_posix(),
    })
    prof = ET.SubElement(mlt, "profile", PROFILE)

    bg = prods.color("background_producer", "#ff000000", total_f)

    for p in prods.elements:
        mlt.append(p)

    bg_pl = _playlist("background")
    ET.SubElement(bg_pl, "entry", {"producer": bg, "in": "0", "out": str(total_f - 1)})
    mlt.append(bg_pl)

    main_bin = ET.Element("playlist", {"id": "main_bin"})
    ET.SubElement(main_bin, "property", {"name": "kdenlive:docproperties.decimalPoint"}).text = "."
    ET.SubElement(main_bin, "property", {"name": "kdenlive:docproperties.version"}).text = "1.1"
    ET.SubElement(main_bin, "property", {"name": "kdenlive:docproperties.profile"}).text = "atsc_1080p_30"
    ET.SubElement(main_bin, "property", {"name": "xml_retain"}).text = "1"
    for p in prods.elements:
        if p.get("id", "").startswith("producer"):
            ET.SubElement(main_bin, "entry",
                          {"producer": p.get("id"), "in": p.get("in"), "out": p.get("out")})
    mlt.append(main_bin)

    for pl in (a1, a2, v1, v2, v3):
        mlt.append(pl)

    tractor = ET.SubElement(mlt, "tractor", {
        "id": "tractor0", "title": title, "global_feed": "1",
        "in": "0", "out": str(total_f - 1),
    })
    ET.SubElement(tractor, "property", {"name": "kdenlive:projectTractor"}).text = "1"
    ET.SubElement(tractor, "track", {"producer": "background", "hide": "audio"})
    ET.SubElement(tractor, "track", {"producer": "playlist_a1", "hide": "video"})
    ET.SubElement(tractor, "track", {"producer": "playlist_a2", "hide": "video"})
    ET.SubElement(tractor, "track", {"producer": "playlist_v1", "hide": "audio"})
    ET.SubElement(tractor, "track", {"producer": "playlist_v2", "hide": "audio"})
    ET.SubElement(tractor, "track", {"producer": "playlist_v3", "hide": "audio"})

    n = 0
    tractor.append(_transition(f"transition{n}", "mix", 0, 1)); n += 1
    tractor.append(_transition(f"transition{n}", "mix", 0, 2)); n += 1
    for d in edl.dissolves:
        span = d.get("actual", d["dur"])
        if span <= 0.02:
            continue
        a = frames(d["at"])
        b = max(a + 1, frames(d["at"] + span) - 1)
        tractor.append(_transition(f"transition{n}", "luma", 3, 4,
                                   always=False, in_f=a, out_f=b)); n += 1
    tractor.append(_transition(f"transition{n}", composite, 0, 3)); n += 1
    tractor.append(_transition(f"transition{n}", composite, 0, 5)); n += 1

    if _overlaps and log is not None:
        log.append(f"project: WARNING {len(_overlaps)} clip(s) could not start where "
                   f"the EDL says - {_overlaps[:3]}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(mlt, space="  ")
    xml = ET.tostring(mlt, encoding="unicode")
    out_path.write_text('<?xml version="1.0" encoding="utf-8"?>\n' + xml + "\n",
                        encoding="utf-8")
    return out_path


def _filter(parent: ET.Element, service: str, props: dict) -> ET.Element:
    f = ET.SubElement(parent, "filter", {"id": f"f{id(parent)}_{service}"})
    ET.SubElement(f, "property", {"name": "mlt_service"}).text = service
    for k, v in props.items():
        ET.SubElement(f, "property", {"name": k}).text = str(v)
    return f


