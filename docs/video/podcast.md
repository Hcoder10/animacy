# The podcast set

Two hosts on a dark studio set: the **Autonomous Lamp** and the **Reachy Mini**,
reading `docs/video/script.md` to each other. Every joint value in the footage
came out of `animacy` from the audio of that same line. Nothing in this pipeline
is keyframed, and there is no animation curve anywhere in it.

```
docs/video/script.md ─┐
data/video/voice/     ├─► scripts/video/show_build.py ─► data/video/podcast/show.json
  manifest.json + wavs┘                                   data/video/podcast/narration.wav
                                                                    │
                            web/podcast.html + web/js/podcast*.js ◄──┘
                                        │  window.podcast.seek(i)
                                        ▼
                       scripts/video/podcast_render.py ─► data/video/podcast/<cam>/<section>.mp4
                                                          data/video/podcast/render_manifest.json
```

## Where the motion comes from

For every line, the **speaking** host's motion is
`animacy.serve.retrieval_motion(wav, sr, checkpoint="checkpoints/v2a", intent=<the line>)`
— motion matching against the human corpus, driven by that line's own waveform —
then `animacy.retarget.retarget_clip` through that robot's `ROBOT.md`.

The **listening** host gets the same call with `listen=True`, which is the causal
path with `speaking = 0`: listening is its own behaviour in the model, not a
paused talker. On top of it goes the viewer's listen-mode gaze overlay — a
constant look direction toward whoever is speaking, blended under the generated
motion exactly as `web/js/talk.js` does it:

```
head_yaw   += GAZE_WEIGHT · g_yaw          GAZE_WEIGHT = 0.5
head_pitch += GAZE_WEIGHT · g_pitch        then the canonical sanity bounds
```

`g_yaw` differs per host (`show_build.py: HOSTS[...]["gaze_yaw"]`, 16 deg for the
lamp, 28 for the reachy) because the same canonical degree buys a very different
amount of *looking* on each body. The lamp's `head_yaw` drives `base_yaw` at gain
-1.37 **and** `wrist_roll` at -1.99, and rolling a shade whose optical axis is
`[0.7, 0, -0.71]` in the head frame swings its gaze as well. Measured, the lamp
turns 3.05 deg of gaze per canonical degree of `head_yaw` against the reachy's
1.57 — about twice as far — which is why the two numbers are not the same.

**The sign of that offset is a measured fact, not a derived one.** Reading it off
the mapping gains got it wrong for *both* robots on the first pass — and the two
hosts need *opposite* canonical signs even though they face each other, because
the lamp's `head_yaw` gain is negative and the reachy's is positive.
`window.podcast.measure(frame)` reports each host's gaze angle away from pointing
straight at the other; a listening host's number must go **down** relative to
rest. Currently measured over all 34 lines:

| host | rest | listening (median) | |
|---|---|---|---|
| LAMP | 74.1 deg | **49.7** | OK |
| REACHY | 37.9 deg | **15.9** | OK |

(Re-measured on the current 6719-frame cut: LAMP 74.1 -> 48.6, REACHY 37.9 ->
16.1. Re-measure after any change to the placement, either `ROBOT.md`, or the
gaze constants.)

Re-run that check if either `ROBOT.md`, either URDF, or the placement changes.

The only hand-authored numbers in the whole show are the set itself: the settle
between lines, the beat between sections, and how far each host is turned.

## The show clock

`show_build.py` puts audio and motion on one integer frame clock at 30 fps, so
they cannot drift:

| | frames | seconds |
|---|---|---|
| lead-in (both at rest) | 36 | 1.20 |
| settle between lines in a section | 11 | 0.367 |
| beat between sections (ease to rest, hold, ease out) | 16 | 0.533 |
| tail | 45 | 1.50 |

The section beat is 16 frames because the edit caps section gaps at 0.55 s; making
it fit means the cut never has to trim one. The in-section settle is deliberately
*not* shortened — both robots are mid-behaviour there, and trimming would slide
the picture against the motion generated for it.

Line *k* starts at row `f_start` of the joint tracks and at sample
`f_start / 30 · 16000` of `narration.wav`. Gaps are a smoothstep from the last
pose of one line to the first pose of the next; section gaps pass through the
robot's profile rest pose and hold it, which is the "attentive rest" beat.

`retarget_clip` stretches time wherever a joint would break its speed ceiling, so
its table can run a frame or two past the audio. Each line's frame count is the
max over both robots and the audio, and the tail of a shorter table is held — so
no motion is ever cut and the pose simply rests into the gap.

### `show.json`

```jsonc
{
  "fps": 30, "n_frames": 6719, "seconds": 223.97,  // 34 lines, 9 sections
  "narration_wav": "narration.wav",
  "placeholder_voice": false,
  "hosts":    { "lamp": {"robot": "lamp", "joints": [...], "rest": [...]}, "reachy": {...} },
  "tracks":   { "lamp": {"joints": [...], "values": [[...], ...]},  // n_frames rows, profile units
                "reachy": {...} },                                  // (deg / mm, as ROBOT.md declares)
  "lines":    [ {"index", "section", "host", "text", "wav", "f_start", "f_count",
                 "t_start", "seconds", "lamp": {joint table}, "reachy": {joint table}} ],
  "sections": [ {"index", "title", "line_indices", "f_start", "f_end", "t_start", "t_end"} ]
}
```

`tracks` is the whole show including the gaps — both robots have a value for
every frame — and is what the player reads. `lines[].lamp/.reachy` are the
per-line tables before the gaps were inserted.

## The set

`web/js/podcast_set.js`. A dark studio: a matte floor, a cyclorama with a cool
pool of light on it for separation, and two plinths.

- **Placement.** The hosts sit on a line oblique to the lens — the lamp nearer
  and camera-left, the reachy further and camera-right. Side by side, an
  over-the-shoulder is geometrically impossible; oblique, it works and the wide
  gains some depth.
- **Eyeline.** `EYELINE = 0.56 m`. Each plinth's height is *solved* so that
  robot's head link lands there, measured in the rest pose — 0.241 m under the
  lamp, 0.368 m under the (shorter) reachy. That is why they differ.
- **Facing.** `PLACE[host].gazeAz` is where the host's **gaze** points, ±24 deg
  off the lens. It is not a body yaw: a lamp's shade does not point along its
  URDF's +x, so `podcast.js` measures the rest gaze direction (via
  `description.viewer.gaze`) and solves the body yaw that lands it there.
- **Light.** Warm key from camera-left, above and in front — the only shadow
  caster. Cool rim from behind camera-right. A low neutral fill from camera-right
  (no shadows of its own: a second shadow caster on a two-hander reads as a
  mistake), a warm bounce so the plinths are not holes, and a wash on the
  cyclorama placed past the hosts so it cannot spill onto them.
- **Why the key is only 10.5.** Both shells are near-white, and the lamp's base
  carries a fan grille and vent slots whose upward faces sit directly under the
  key. Any brighter and they clip to flat white and read as *shattered geometry*.
  That is what the "white shards on the lamp base" were — not a broken or badly
  decimated mesh (there is one `base.stl` and no decimated variant), and not
  normals. Exposure 0.95 and roughness 0.52 hold the same highlight.

### Shading: why `computeVertexNormals()` alone cannot help

An STL stores three **unshared** vertices per triangle. The viewer's
`computeVertexNormals()` therefore has nothing to average across and can only
ever produce flat shading — which reads as hard facets on the lamp's shade and
Reachy's shell at 1080p. No smoothing-angle setting fixes that, because there are
no shared vertices to smooth over. `podcast.js` welds the duplicates with
`mergeVertices()` first, then `toCreasedNormals(geometry, 42°)`, so curved
surfaces shade smoothly and genuine edges stay hard.

Shells are also drawn `DoubleSide` (the viewer's choice: exported STLs mix
windings). That is fine until they cast shadows — back faces get written into the
shadow map and a part with interior geometry self-shadows into hard artefacts —
so the podcast page sets `material.shadowSide = THREE.FrontSide`.

Because the plinth heights, the body yaws and every camera are *solved from
measurements of the loaded URDFs*, the set survives a robot changing shape. If a
`ROBOT.md` rest pose moves, re-run `--probe` and look; nothing needs re-typing.

`--probe` also prints each robot's motion box, which is the clipping check: over
the whole show the two boxes overlap by **0.000 m** on z (lamp z ∈ [0.068,
0.365], reachy z ∈ [-0.327, -0.066]), and the plinths are 0.481 m apart against a
radius sum of 0.280 m. Neither robot can reach the other or its plinth.

## The cameras

Five static tripods, `?cam=A..E`. Each is fitted against the union of each
robot's bounding box over **every frame of the show** — not its rest pose — so a
raised shade or a swung antenna cannot leave frame halfway through a take.

| | shot |
|---|---|
| A | wide two-shot, the whole show |
| B | single on LAMP, fills frame |
| C | single on REACHY, fills frame |
| D | over the shoulder, behind LAMP onto REACHY |
| E | the wide with a 1 deg push-in across the take (open and close) |

A is one file over the whole timeline from t=0. B/C/D/E are one file per section,
`<cam>/sNN.mp4`, NN being the 1-based section number; E additionally has
`open.mp4` (lead-in + section 1) and `close.mp4` (section 9 + tail).

`--tail-frames N` extends each per-section take past its section end so a cut can
run into the following gap instead of freezing on the last frame. It never moves
a clip's **first** frame, so `t_start` — the only thing an edit needs to sync on
— is unaffected. 30 frames (1.0 s) is the useful default; without it, a shot that
runs past its section has nothing to dissolve out of.

`--suffix _v2` writes `<section>_v2.mp4`, so a re-render never overwrites a file
an editor is already reading. Every manifest entry carries a `variant`, and the
manifest's `convention` block states all of the above rather than implying it.

No orbiting, no bobbing, no handheld. E's push is the only camera move: it loses
`push` degrees of field of view linearly across the frames of *that take*, which
is why the renderer passes `f0`/`f1` when it sets the camera.

## The player

`web/podcast.html` has no clock. `window.podcast.seek(i)` applies row *i* of both
tracks and renders once — frame *i* of a video is row *i* of the show, whatever
frame rate the machine drawing it manages. `window.podcast.ready` resolves when
both URDFs and the show are in.

```js
await window.podcast.ready;
window.podcast.seek(1200);        // apply frame 1200, render
window.podcast.setCamera('D', 900, 1800);
window.podcast.measure(1200);     // gaze geometry at that frame
window.podcast.debug();           // plinths, heads, boxes, camera rigs
```

The URDF load and the joint application are the viewer's, not a copy of them:
`RobotViewer` (`web/js/viewer.js`) parses the URDF and owns `setJoints`, and the
profile-units-to-URDF maths is `toUrdfValues` from `web/js/retarget.js`. The one
thing `podcast.js` does differently is where the robot ends up — each is
re-parented out of its own viewport into one studio scene, under the same
z-up-to-y-up rotation the viewer uses.

## Running it

```bash
# 1. the show (needs animacy: the hermes python has it)
python scripts/video/show_build.py

# 2. look before you render (Playwright lives in reachy-duplex/.venv)
C:/Users/sarta/reachy-duplex/.venv/Scripts/python.exe scripts/video/podcast_render.py --probe
C:/Users/sarta/reachy-duplex/.venv/Scripts/python.exe scripts/video/podcast_render.py --stills
C:/Users/sarta/reachy-duplex/.venv/Scripts/python.exe scripts/video/podcast_render.py --bench 60

# 3. render
C:/Users/sarta/reachy-duplex/.venv/Scripts/python.exe scripts/video/podcast_render.py --cam A
C:/Users/sarta/reachy-duplex/.venv/Scripts/python.exe scripts/video/podcast_render.py   # everything
```

Runs headed by default, because headed Chromium gets a real GPU context — on this
box `ANGLE (NVIDIA RTX 5080, D3D11)`, which grabs 1920x1080 frames at ~24 fps, so
the whole 30-take set is about 20 minutes. Headless falls back to SwiftShader and
is roughly twenty times slower at this resolution. If there is no display to open
a window on, the renderer says so and drops to headless by itself; `--headless`
forces it when you want the machine back.

`render_manifest.json` is written after every clip and merges with what is
already in it, so a take that fails can be re-run on its own with `--cam` /
`--section` without disturbing the rest. `--limit-frames N` is a smoke test that
writes a truncated take *to the real path* — re-render that camera without it
afterwards.

Every clip carries the matching slice of `narration.wav`, not only the wide.
Since the show clock *is* the frame clock, the slice for frames `[f0, f1)` is
exactly `f0/30` for `(f1-f0)/30` seconds, so any two angles cut together stay in
sync on picture and on sound.

## If the voice takes are not ready

`show_build.py` falls back to local SAPI TTS (a different installed voice per
host) and writes `placeholder_voice: true` into `show.json`; the renderer prints
a warning when it sees it. The motion is real — it is generated from that
placeholder audio — so the set, the framing and the timing are all workable. It
is not the footage: re-run `show_build.py` when
`data/video/voice/manifest.json` lands, then re-render.

## One manifest, one timeline

The show was rebuilt once mid-production (a reworded line, plus shorter section
gaps), which left two generations of footage on disk at different lengths. That
is a live hazard: an edit reads `t_start` out of `render_manifest.json`, so a
clip from the wrong generation sends every cut on that camera to the wrong frame
— and the mismatch is invisible in a still and obvious the moment anyone watches.

So `render_manifest.json` describes exactly one timeline. It carries a
`timeline` block naming the show, its frame count and its length, and it lists
only clips that fit it; superseded clips move to `render_manifest_v1.json`
stamped SUPERSEDED, beside the `show_v1.json` they belong to. Re-renders go to a
`--suffix` rather than over a file someone may be reading.

If you rebuild the show, the old renders do not become slightly wrong — they
become a different film. Split them immediately.

### Checking the show still matches its audio

Voice takes get re-rendered. Before trusting a build, compare every wav on disk
against `show.json`:

- each line's duration against its `audio_seconds` — drift beyond one frame
  (33.3 ms at 30 fps) means the motion no longer sits under the words;
- each line's `f_count / fps` against the wav — the motion must still *cover*
  the audio, since a line's frame count is the max over both robots and the
  audio, and a longer wav would run past the animation.

At the time of writing, a re-render for bit-reproducibility left all 34 lines at
**0.0 ms** drift, so the build stayed valid without touching it.

`data/video/` is gitignored. Nothing here is committed.
