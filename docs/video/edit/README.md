# The edit

How the demo film is assembled, how to re-render it, and where every frame
came from.

The whole cut is generated from one command. Nothing is hand-placed on a
timeline, but the *decisions* — which angle, where the cuts land, what gets a
dissolve — are written down explicitly in `scripts/video/edit_common.py` rather
than left to chance, and the result is a real MLT project you can open and
tweak by hand.

```
python scripts/video/edit_all.py
```

Re-runnable at any point. Whatever footage exists is cut in; whatever is
missing becomes a labelled placeholder card in the timeline and is listed in
the run report, so this can be run the moment the first line of narration
lands and again after every delivery.

## Deliverables

| File | What it is |
|---|---|
| `docs/media/animacy_demo.mp4` | the film, 1920x1080 h264/yuv420p, AAC 192k, faststart |
| `docs/media/animacy_demo_720.mp4` | 1280x720 cut-down for the README and social |
| `docs/media/animacy_lamp_loop.mp4` | 12 s silent loop of the lamp's retargeted nod, for the README header (see below) |
| `docs/video/edit/animacy_demo.kdenlive` | the MLT project — open it in Kdenlive or Shotcut |
| `data/video/edit/edl.json` | the full edit decision list: every shot, cue, title and transition |
| `data/video/edit/report.json` | what the last run produced, including anything still missing |
| `data/video/edit/qc/` | frames pulled back out of the finished render and looked at |

## How the cut is structured

**The narration is the spine.** Every video cut lands on the end of a spoken
line. Line timings are not invented by the edit — they are adopted wholesale
from `data/video/podcast/show.json`, because the two robot hosts' motion is
generated from the narration on that clock and baked into the camera renders.
If the edit used its own timing the robots would drift against their own
voices. Each shot pulls source frames at

```
src_in = line.t_start + offset_into_the_shot - clip_start_time
```

so a J-cut or a mid-line cut still lands on the right frame of the
performance. Gaps *inside* a section are left exactly as rendered for the same
reason; the only gap the edit shortens is the one at a section boundary, which
is always a cut to a different source anyway.

**Angles.** The wide two-shot (A) carries exchanges and every section's
punchline. Single lines go to the lamp single (B) or the reachy single (C).
The over-shoulder (D) is used once, in section 3, on "my head is a six
degree-of-freedom Stewart platform. His is a lamp arm on a bearing." The slow
push-in (E) opens the film on the cold open and closes it on the last line.
The mapping is the `CUTPLAN` table in `edit_common.py`, one entry per line of
the script, keyed on the line's position *as written* so that dropping a line
for length does not slide every later shot onto the wrong camera.

A per-section host clip stops at its section end, so it cannot cover a shot
that runs on into the gap; holding its last frame there would freeze a talking
robot mid-word. When that happens the build silently prefers another angle
that does have the frames — usually camera A, which is rendered as one
continuous file — and notes the substitution on the shot.

**B-roll** covers the explanations — the terminal, the ROBOT.md, the viewer,
the dataset report, the grading table — and the picture returns to the wide for
the line that closes each section. Slots are matched to whatever the b-roll
agent actually delivered by keyword and by the `section` tag in
`data/video/broll/manifest.json`; when one clip has to serve two slots the
in-point advances instead of replaying the same frames. Portrait clips are
penalised so they do not get pillarboxed into a 16:9 frame.

**Transitions.** 31 hard cuts and 3 dissolves, at roughly 16 s, 72 s and
140 s.

A dissolve spans `[T, T+d]` — the *incoming* picture starts on time at its
natural in-point and the *outgoing* clip is what stretches across and fades
away. Doing it the other way round would slide the incoming shot half a second
ahead of its own audio for the whole shot.

Where the dissolves go was decided by looking at the render, not by taste
alone. All five cameras point at the same two robots on the same locked-off
set, so dissolving one host angle into another double-exposes the robots — two
lamps, two Reachys, ghosted over each other — and reads as a fault rather than
a transition. The first cut had dissolves at the section 1→2, 4→5 and 8→9
boundaries and two of the three ghosted visibly. They now land only where the
picture changes character, host shot into b-roll: into *capture* (section 2),
into *retarget* (section 4), and into *on real hardware* (section 6).
`build_edl` additionally refuses any dissolve whose two shots both resolve to
host cameras, and logs the refusal, so a change of footage cannot quietly
reintroduce the ghosting.

**J-cuts** at three boundaries: the next section's narration starts 0.50–0.60 s
before its picture arrives (sections 3, 7, 8).

**Lower-thirds.** Five, one per idea: *capture*, *one file per robot*,
*retarget*, *the interaction layer*, *on real hardware*. Small, static, bottom
left, Inter at 34 px with a 2 px rule and a soft wash for legibility. They
appear on a cut, hold 2.5 s, and fade in and out over 6 frames. Nothing else
moves.

**Open and close.** The film opens cold on the first exchange with no title at
all. The only card is the end card: `animacy`, the repo URL, the viewer URL,
held 3 s, then black.

**Audio.** Narration at −16 LUFS integrated, measured across the whole
performance and corrected with a single static gain so the deadpan delivery is
not flattened by a compressor. B-roll is silent by design. Under the host
scenes only, there is generated pink noise at −50 dBFS as room tone, fading in
and out at each host run. **There is no music.**

## The lamp loop

`docs/media/animacy_lamp_loop.mp4` is the lamp doing the retargeted nod, A/B
against the vendor's hand-made version, 12 s, silent, 1280x720, with a short
fade at the seam so it cycles cleanly.

Its source has to be that shot and nothing else. An earlier run picked
`s4_retarget_csv.mp4` — a terminal recording — purely because "retarget" was in
its filename, and a README header that claims to show the lamp moving while
showing scrolling text would misrepresent the project. The selector now needs a
positive match on lamp/viewer/nod/lean/gaze **and** a clean pass against a
blocklist of terminal, table, report, score and readback clips. If nothing
qualifies it produces no file and says so in the run log, rather than shipping
the wrong thing.

## Length

The film must land between 2:30 and 3:30. The full script as recorded runs
longer than that, so `DROP_ORDER` in `edit_common.py` lists the lines that may
be cut, worst-first, and the build drops from that list until the film fits.
The order deliberately protects the two things that make the film credible —
the SO-101 proof point, and the admission that the learned model loses on beat
alignment — so those go last, and the learned-model pair goes together or not
at all.

Dropping a line leaves a join inside a continuous host render, which would
read as a jump cut. The build detects those joins and forces a change of angle
across them, which is what an editor would do by hand.

Whatever was dropped on a given run is listed in `report.json` under
`dropped_lines`, so it is never a silent edit.

## Rendering

The default engine is **melt**, from MLT — the same library Kdenlive and
Shotcut are built on. It renders `data/video/edit/animacy_demo.mlt`, which is
the same timeline as the `.kdenlive` project, differing only in which
compositing service it names: the project uses `frei0r.cairoblend` because
that is what Kdenlive expects, and the render copy uses whatever the local
melt build actually reports via `melt -query transitions`.

If melt is missing or its render fails, `edit_render.render_ffmpeg` assembles
the identical EDL with an ffmpeg `filter_complex` — per-shot intermediates,
`concat` within a run, `xfade` at each dissolve, `overlay` for the
lower-thirds. Either way the `.kdenlive` project is written, so the edit can
always be opened and adjusted by hand.

`report.json` records which engine produced the file.

MLT is not on PATH by default on Windows. The build finds `melt.exe` under
`C:/Users/sarta/mlt-portable/Shotcut/` (the Shotcut portable zip, which needs
no installer — the Kdenlive installer wants elevation) or anywhere else on
PATH. To point it somewhere else, edit `find_melt()` in `edit_common.py`.

## Watching our own output

Rendering something is not the same as it being right, so every run pulls
frames back out of the finished file and checks them:

- 15 frames extracted to `data/video/edit/qc/`, deliberately at the moments
  that matter — every lower-third, the midpoint of every dissolve, the end
  card, the opening frame — rather than evenly spaced and hoping.
- `blackdetect` and `freezedetect` over the whole master, with the placeholder
  cards and the end card excluded, so a clip that silently rendered black or
  froze shows up as a warning instead of being shipped.
- Both file size ceilings are enforced by re-encoding at a computed bitrate if
  the quality-targeted pass overshoots.

## Where the material comes from

| Source | Owner | Used for |
|---|---|---|
| `data/video/voice/manifest.json` + WAVs | `voice` | every spoken line |
| `data/video/podcast/show.json` | `podcast` | **the master clock** — line timings, section bounds |
| `data/video/podcast/<cam>/*.mp4` | `podcast` | the two hosts, cameras A–E |
| `data/video/broll/*.mp4` + `manifest.json` | `broll` | terminals, viewer, dataset, read-back, grading |
| `docs/video/script.md` | — | section structure and every line of dialogue |

Camera clips may be shipped either as one file per section
(`<cam>/<NN>.mp4`, starting at that section's `t_start`) or as one long file
per camera (`<cam>/full.mp4`, starting at t=0). Both are handled. What matters
is that `render_manifest.json` states, for each clip, the podcast-timeline time
its first frame corresponds to — that number is what keeps the cuts in sync.

Generated media — lower-thirds, the end card, placeholder cards, room tone,
levelled narration — lands in `data/video/edit/gen/` and is reproduced from
scratch on every run. It is safe to delete.

## Options

```
python scripts/video/edit_all.py --no-render          # EDL + project only
python scripts/video/edit_all.py --engine ffmpeg      # skip melt
python scripts/video/edit_all.py --max-runtime 195    # tighter ceiling
python scripts/video/edit_all.py --skip-derivatives   # master only
```
