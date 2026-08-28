# Narration voices for the demo film

The two hosts in [`docs/video/script.md`](../script.md) are spoken by
**Kokoro-82M**, run locally. Everything under `data/video/voice/` is generated
by [`scripts/video/tts_render.py`](../../../scripts/video/tts_render.py) and is
not in git (`data/video/` is gitignored) — regenerate it with one command.

```bash
python scripts/video/tts_render.py --script docs/video/script.md --out data/video/voice
```

34 lines, **204.1 s** (3 min 24 s) of speech.

## The voices

| Host | Voice | Accent | Median f0 | Pace | Speed |
|---|---|---|---|---|---|
| **LAMP** | `bm_lewis` | British male | 94.9 Hz | 3.75 syl/s | 0.90 |
| **REACHY** | `am_michael` | American male | 116.5 Hz | 4.25 syl/s | 1.03 |

Both figures are measured over the delivered audio, not claimed from the model
card. REACHY sits **22.8 % higher** in pitch and **13 % quicker**, on top of a
different accent — so the two hosts separate on three independent axes and stay
distinguishable even under B-roll, where the picture is not on the speaker.

The script refers to both hosts as "he" ("watch me while *he* talks", "*he* does
not have a body yet"), so both voices are male.

`speed` is Kokoro's own duration scale — it stretches the predicted phoneme
durations and leaves pitch alone. Nothing is resampled or pitch-shifted, so
neither voice has the chipmunk / slowed-tape artefacts that speed-changing a
finished waveform would give.

### Why these two

Ten male candidates were rendered on real script lines and scored on measured
pitch, measured speaking rate, and Whisper word-error rate — the practical test
for the thin, wobbly artefacts the smaller-data voices in this pack can produce.
All ten were intelligible, so the choice came down to character fit and
separation.

`bm_lewis` measured ~95 Hz, the lowest of the ten except `am_onyx` (~87 Hz,
but American and naturally faster, so it read as brisk rather than deadpan).
Slowed to 0.90 it is dry and deliberate, which is the brief for LAMP. Kokoro's
own voice table marks it as one of the thinner-trained voices, so it was the
candidate most at risk of artefacting — it transcribed at **0 % WER** on one of
the densest technical lines in the script, so the risk did not materialise.

`am_michael` measured ~118 Hz, warm and mid-range, and also transcribed clean.
Blends were tried — Kokoro averages style vectors for a comma-separated voice
list — but every blend that stabilised LAMP pulled its pitch up towards REACHY's
(a 50/50 `bm_george,bm_lewis` landed at 120 Hz, on top of REACHY) and cost more
separation than it gained. Both hosts therefore use a single unblended voice.

## Audio format

```
data/video/voice/
  manifest.json                     see below
  NN_<host>_<slug>.wav              masters — 24 kHz mono PCM16
  16k/NN_<host>_<slug>.wav          exact 16 kHz mono PCM16 copies
  _raw/                             pre-loudnorm intermediates, ignorable
```

The masters are 24 kHz because that is Kokoro's native output rate — writing
them at 16 kHz would throw away detail before the edit needs to. The `16k/`
copies are there for anything that wants
[`animacy.schema.AUDIO_SR`](../../../animacy/schema.py) without resampling;
`scripts/video/show_build.py` reads the 24 kHz masters and resamples them
itself, so it needs neither.

Each line is silence-trimmed to a 40 ms head/tail pad with 5 ms edge fades (so
no onset clicks), then loudness-matched with two-pass ffmpeg `loudnorm` to
**-18 LUFS** with a **-1.5 dBTP** ceiling. Delivered peaks land between -1.43
and -1.78 dBFS with zero clipped samples, so no line jumps out in the edit and
there is headroom for the mix.

## manifest.json

A JSON object — the per-line array is `manifest["lines"]`, and the top level
carries the engine, voice and audio metadata.

```jsonc
{
  "engine":     { "engine": "kokoro", "repo_id": "hexgrad/Kokoro-82M",
                  "licence": "Apache-2.0", "device": "cpu", ... },
  "voices":     { "lamp": {...}, "reachy": {...} },
  "audio":      { "sample_rate": 24000, "channels": 1, "format": "pcm_s16le",
                  "loudness_target_lufs": -18.0, "true_peak_ceiling_dbfs": -1.5 },
  "index_base": 0,
  "line_count": 34,
  "total_seconds": 204.06,
  "lines": [
    { "index": 0,                   // 0-based, script order
      "host": "lamp",               // "lamp" | "reachy"
      "section": "1 — Cold open",
      "text": "...",                // verbatim from script.md
      "text_spoken": "...",         // after TTS normalisation, see below
      "wav": "01_lamp_....wav",     // relative to the manifest's directory
      "wav16k": "16k/01_lamp_....wav",
      "seconds": 5.45, "voice": "bm_lewis", "speed": 0.9, "peak_dbfs": -1.43 }
  ]
}
```

`index` is **0-based** to match `scripts/video/show_build.py`, which numbers its
own script lines from zero. The `NN_` filename prefix is 1-based and exists only
so the takes sort in script order — `file_number == index + 1`.

## Text normalisation

The synthesiser is fed `text_spoken`, not `text`. Every substitution was chosen
by reading Kokoro's phonemiser output, not by guessing, and the manifest keeps
the verbatim script line alongside it.

| Script | Spoken | Why |
|---|---|---|
| `README` | `read me` | otherwise spelled out, R-E-A-D-M-E |
| `degree-of-freedom` | `degree of freedom` | the hyphens collapse it into one run-on word |
| `Retarget` | `Re-target` | otherwise "ri-TAR-git"; the field says "REE-target" |

Em dashes carry the performance in this script, so they are translated rather
than deleted. An internal `—` becomes a comma. A **trailing** `—` (REACHY being
cut off at the end of section 5) drops the dash *and* any final full stop, so
the line ends on open, unfinished intonation. A **leading** `—` (LAMP finishing
REACHY's sentence) is dropped and the sentence re-capitalised.

Words that a technical script usually breaks were checked and left alone because
Kokoro already gets them right: *animacy* (ˈanɪməsi), *SO-101* ("S O one oh
one"), *CSV*, *Autonomous OS*, *Stewart*, *hertz*, *daemon*, *kinematics*.

## Reproducibility

**Durations are stable; bytes are not guaranteed identical.** This is the
distinction that matters, because the film's timeline is built from the
durations.

On the default `--device cpu`, every re-run produces WAVs with **identical
sample counts** — so every `seconds` value, and any timeline built from them, is
reproducible exactly. The samples themselves can differ by a few LSB: measured
across a full re-run, the worst per-sample difference was **4/32768, about
-78 dBFS**, far below the 16-bit noise floor and inaudible. Two back-to-back
runs were byte-identical; a later run on the same machine under different load
was not, which points at CPU LSTM reductions varying with thread scheduling
(torch used 24 threads here). Pin `OMP_NUM_THREADS=1` if you need the bytes to
match as well as the durations.

`--device cuda` is roughly twice as fast (36 s against 84 s for the whole
script) but is not duration-stable: an earlier GPU pass and its repeat differed
on 12 of 34 files and moved total speech length by 1.5 s. That is the difference
that would desync an edit, so CUDA is opt-in and the shipped audio is the CPU
render.

### The film's timeline was built from an earlier render

The cut that ships — `show.json`, `narration.wav` and the rendered camera takes
— was built from an **earlier pass** of these voices, not from the files
currently on disk. Regenerating with the command above produces takes whose
durations differ from that pass by tens of milliseconds (total speech length
202.6 s then against 204.1 s now, the bulk of it one line in §6 whose wording was
corrected for accuracy after the first render).

So: **the files here are the reproducible set, not the set the film was cut
against.** Re-running `show_build.py` against them would desync the rendered
footage. If the film is ever rebuilt, rebuild it end to end from these files
rather than mixing the two.

Other flags: `--engine {kokoro,sapi}` (`sapi` is a no-download Windows
System.Speech fallback, markedly more robotic — for when you want timings without
waiting on a model), `--voice-lamp` / `--voice-reachy`, `--speed-lamp` /
`--speed-reachy`, `--sr`, `--seed`, `--only N` (0-based), `--no-loudnorm`.

## Verification

Every delivered line was transcribed with `openai/whisper-small.en` and compared
against the script: **all 34 match word-for-word**, with no dropped or truncated
words. Also checked: zero clipped samples, durations agree with the manifest to
within 20 ms, speaking rate 3.4-5.5 syl/s (mean 4.0 — natural conversational
range), and no internal silence longer than 0.68 s (a sentence boundary inside
the longest take).

## Licence

| | |
|---|---|
| Kokoro-82M weights | Apache-2.0 — [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) |
| `kokoro` and `misaki` packages | Apache-2.0 (both, verified from package metadata) |
| Voice packs (`bm_lewis`, `am_michael`) | Apache-2.0, distributed in the model repo |

Apache-2.0 permits commercial and derivative use, so the narration can ship in
the film without a separate voice licence. This matters for a project whose own
pitch is that every clip carries its licence evidence — the narration is held to
the same standard as the motion data.

Kokoro is also already this repo's browser-demo voice
(`animacy.tts.synth_kokoro`), so the film and the live demo speak with the same
engine.

## PersonaPlex was not used — an honest note

The brief's first choice was NVIDIA PersonaPlex-7B on the `squaredcube1` rig.
It was checked and rejected on suitability, not availability:

* **VRAM was not the blocker.** GPU 2 on the rig had 95.9 GB free.
* PersonaPlex is a **full-duplex conversational speech model**. Its job is to
  hold a live two-way conversation, not to read 34 fixed lines verbatim. Making
  it speak an exact script needs the text-forcing/injection machinery from
  `reachy-duplex`, and the known failure mode is drift and garbling on longer
  free generation. Turning that into a reliable batch renderer is a research
  task, and this render had a same-night deadline.
* Kokoro is deterministic and says exactly what it is given — verified above,
  word-for-word on all 34 lines. For scripted narration that is worth more than
  a larger model's expressiveness.

The trade-off is real and worth stating plainly: **Kokoro is good scripted TTS,
not a conversational speech model.** The delivery is clean, correctly stressed
and well separated between the two hosts, but it is a read, not a performance —
the hosts do not truly interrupt or react to each other, and the film's comic
timing lives in the edit rather than in the voices. If PersonaPlex is wanted for
a later cut, the script, the normalisation table and the per-line structure here
all carry straight over; only the `Engine` class in `tts_render.py` changes.
