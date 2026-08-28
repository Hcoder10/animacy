# B-roll for the demo film

Footage for `docs/video/script.md`. Every clip is **real**: real stdout from a
command that actually ran on this machine, the real web viewer, the real
dataset pages, the real robot's own read-back. Nothing is a mock-up, a
re-creation or a motion graphic.

- **Format** — 1920x1080, 30 fps, MP4/h264, **silent** (the edit adds narration).
- **Length** — 6-15 s, except four section-4 clips the editor asked to run long
  (`s4_ab_vendor_nod`, `s4_ab_lamp_hero_loop`, `s4_lean_in`, `s4_speed_cap`, 16-18.7 s):
  spare length gives handles at a dissolve and room to pick the moment.
- **Where** — `data/video/broll/<section>_<slug>.mp4` (gitignored; `data/video/` is in `.gitignore`).
- **Index** — `data/video/broll/manifest.json`: for every clip, the section it
  serves, what it shows, its duration and the exact command or URL that produced it.

Read the manifest first; this page explains how the footage is made and what
the editor should know about each kind of shot.

## Running it

Python: `C:\Users\sarta\reachy-duplex\.venv\Scripts\python.exe`. From the repo root:

| command | shots |
|---|---|
| `python scripts/video/broll_terminal.py` | every terminal + document shot (`--shots` to pick) |
| `python scripts/video/broll_capture.py` | §2 `animacy capture --preview`, screen-recorded |
| `python scripts/video/broll_viewer.py` | §2/§4/§5 the web viewer |
| `python scripts/video/broll_web.py` | §7/§9 Hugging Face, the live site, GitHub |
| `python scripts/video/broll_grading.py` | §8 a slice of a real grading reel |
| `python scripts/video/broll_waveform.py` | §5 speech waveform under the motion it produced |
| `python scripts/video/broll_hardware.py` | §6 registers the supplied phone footage and cuts an excerpt |

Each script re-registers only the clips it makes, so any one can be re-run
without disturbing the rest. Keep to one at a time — they all record.

## How each kind of shot stays real

**Terminal shots** (`broll_terminal.py` → `web/dev/broll/term.html`).
The command runs for real first; its stdout and exit code are captured
verbatim, then replayed in a styled monospace page that only controls the
*pace* at which those bytes appear. The terminal even waits roughly as long as
the real command took before its output starts. Where a shot shows `animacy …`,
the venv's own `animacy.exe` console script is what ran; POSIX one-liners
(`head`, `wc`, `grep`, `ls`) run in Git Bash from the repo root. The type size
is fitted so the widest captured line fills the frame without wrapping — that
is the only presentational choice.

**Document shots** (`web/dev/broll/doc.html`). A real file's text, scrolled at
reading pace or shrunk to fit and held still. The highlighter only wraps spans
around characters that are already there; it never rewrites a line. Each shot
names the file and the line range it shows.

**Viewer shots** (`broll_viewer.py`). The page is parked
(`animacy.setCapture(true)`) and advanced a frame at a time
(`animacy.stepFrame(1/30)`), so frame *i* is exactly *i*/30 s of clip time
however slow the renderer is. Every joint angle comes from the real retargeter
reading the real `ROBOT.md`. Clip time is exact; wall-clock render time is not,
which matters for the Talk shot (see below).

**Screen recording** (`broll_capture.py`). `animacy capture --preview` runs over
a licensed clip and its OpenCV window is grabbed with `ffmpeg -f gdigrab`.
OpenCV's window is DPI-unaware, so on a scaled display the drawn image sits in
the top-left of the reported client area; the script crops to the source
video's own frame size to remove the surrounding desktop.

**Browser shots** (`broll_web.py`). Playwright's own video recorder, in real
time, with the pointer moved in steps and the page scrolled a little at a time.
The pages are live; the only thing the script decides is where the cursor goes.

## `black_windows`: where not to cut

Every manifest entry carries `black_windows` — stretches with nothing on screen,
as `[{start, end, seconds}]` in clip time. An empty list means the clip is clean
end to end. Of the 26 clips only one has any: `s8_grading_reel`, at **0.1-0.6 s**
and **6.3-6.83 s**, the spacers between the judged clips. Keep them (they are
part of what the judge saw) but do not park a cut inside one.

Measured with per-frame peak luma (`signalstats` YMAX below 60, sustained
0.25 s), not with ffmpeg's `blackdetect`. `blackdetect` is wrong for this
footage: these clips are pillarboxed onto a near-black background, so the
padding alone satisfies "most pixels are dark" and a perfectly legible title
card is reported as black — it flags whole dark-but-legible terminal and viewer
clips end to end, and over-reports the reel's second spacer as 6.3-7.83 s when
the "Clip 10" card is already up at 6.9 s. Peak luma asks the question that
matters: is there a bright pixel anywhere?

Regenerate after adding clips with `python scripts/video/broll_annotate_black.py`;
`register()` computes it for new clips automatically.

## Seamless loops

`s4_ab_vendor_nod`, `s4_ab_lamp_hero_loop` and `s4_lean_in` are captured as a
whole number of clip periods from t=0 (6 x 3.00 s, 6 x 3.00 s, 4 x 4.00 s), so
the last frame sits one step before the first and they loop with no cut to hide.
The viewer's camera never moves and there are no cuts inside a take.
`s4_ab_lamp_hero_loop` hides the Reachy viewport and is the intended source for
the README header loop (`docs/media/animacy_lamp_loop.mp4`).

## An encoding bug that was on screen

Commit `9031026` wrote `web/js/main.js` back through a cp1252 round-trip, so
every non-ASCII character in it became mojibake and a BOM was prepended;
`7c028ba` was the last clean revision. This was **visible in the viewer UI** —
`Autonomous Â· 5 joints`, the A/B caption, the demo buttons, Talk status
strings — and it is in footage shot before it was repaired. The file has since
been repaired in the working tree: 49 lines reversed mechanically, 4 lines
(the play/pause/record glyphs, where the round-trip destroyed a byte) taken from
`7c028ba`, BOM removed. Verified three ways: with every run of non-ASCII masked
the repaired file is byte-identical to the damaged one, so no code changed; no
mojibake remains; the result is valid UTF-8.

**Resolved.** The repair was committed and deployed as `36d3c60`. The deployed
file was checked against the working tree by fetching
`https://hcoder10.github.io/animacy/web/js/main.js`: identical, sha1
`069187e30ec7`, zero mojibake, 27 middots and 22 arrows intact.

Every clip in the manifest is clean:

- Only one clip was ever shot inside the damage window (21:56:36 to the repair):
  `s2_channel_bars`, nine seconds after. It is unaffected — the readout's text
  comes from `web/index.html` and the ASCII channel constants, not from the
  damaged strings, and the frames were checked.
- `s4_ab_vendor_nod`, `s4_ab_lamp_hero_loop`, `s4_lean_in` and `s5_talk` were
  re-shot after the repair.
- `s9_live_site` was re-shot after the deploy and now renders
  `Autonomous · 5 joints · urdf/lamp.urdf` correctly.

If you edit `web/js/**`, do not round-trip the files through PowerShell
`Get-Content`/`Set-Content` — that is what caused it (PS 5.1 reads UTF-8 as
ANSI and writes it back as UTF-8). Use Python with `encoding='utf-8'`, or the
editor tool.

## Notes the edit should know

- **§2 channel bars** — the readout is a wide band (about 5.6:1), so the frame
  is that band scaled to full width and centred on the film's background. Crop
  or overlay it as you prefer.
- **§4 speed cap** — `scripts/retarget_eval.py` over the whole corpus (96 human
  clips, 581,923 frames, plus the 31 vendor clips), both robots: speed and limit
  violations are 0 offline and 0 live. The head line naming the corpus is on
  screen above the tables, so the "every clip" claim is scoped where you can see it.
- **§5 Talk** — the line is typed, the button is pressed, and the page's own
  pipeline runs: Kokoro-82M in the browser → speech features → motion
  retrieval → both robots. Kokoro runs on wasm here (Playwright's Chromium
  reports a WebGPU adapter it cannot open a device on), which takes about 12 s
  of wall time to synthesise; because the page is stepped a frame at a time,
  that wait is not in the clip. The motion and its clock are real; the render
  wall time is not the clip's time.
- **§5 waveform** — both traces come from the same graded clip: the audio is
  the TTS the grader synthesised, the joint curves are what `animacy` generated
  from it. The reel part starts with a 1 s title card, which is trimmed so the
  two share a zero. The speech ends before the motion does — that tail is the
  settle, and it is real.
- **§6 read-back** — the sim-to-real replay reports honestly that the head and
  body axes track within a few degrees while the **right antenna reads 61.4° of
  90° commanded** at the brow peak. That outlier is in the published evidence
  table too. If the narration's "every axis read back within a couple of
  degrees" is meant to cover antennas, it needs a tweak.
- **§6 live poll** — the ambient loop rests between clips, so the poll runs 45 s
  and prints the busiest 4.5 s of it. Every row is a real sample at its real
  timestamp and the on-screen header says exactly that.
- **§8 grading reel** — the "Clip N" cards stay in frame on purpose: the card
  gives the spoken line and nothing else, no robot, no source, no method. That
  anonymity is the test. Window offsets come from the run's own sealed manifest
  and the reel builder's `CARD_SECONDS`/`GAP_SECONDS`, so the cut lands on a
  card boundary.
- **§8 score table** — the shot shows animacy **above** the vendor on
  excitement (8.0 vs 6.0) and **below** on thinking (4.0 vs 8.0). That is the
  "level on some movements and below on others" line, on screen, unedited.
- Two clips are not 16:9 in origin and are pillar/letterboxed on the film's own
  background (`#0e1117`): the grading reel (512x512) and the channel band.
- **Footage this pipeline did not make.**
  `real_reachy_desk_16x9.mp4` and `real_reachy_desk_portrait.mp4` are phone
  footage of the physical Reachy Mini, filmed by a person and placed in
  `data/video/broll/` for the edit. They are in the manifest so the edit knows
  they exist, are kept **untouched** (33.6 s each, so outside the 6-15 s spec
  that applies to generated clips), and are marked as supplied. Their
  provenance is only what the file shows — do not present them as evidence of a
  specific run without checking. `s6_reachy_hardware.mp4` is a 12 s excerpt of
  the 16:9 take, conformed to the film's format; its window was chosen by
  frame-difference energy rather than by eye.

## Artifacts written outside `data/video/`

`out/lamp_obama.csv` and `out/data_report.txt` are produced by the shots that
show them, and `out/broll_capture_preview/` is the clip the preview run writes.
`out/` is untracked scratch.
