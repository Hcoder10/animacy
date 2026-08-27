# Large-scale harvest (`scripts/harvest/`)

Goal: >= 5,000 hours of license-clean, single-speaker talking-head human motion in the canonical
clip format (`docs/CANONICAL.md`), harvested continuously and pushed to the Hub in shards. This is
the industrial version of last night's `data_fetch_more.py` / `data_capture_batch.py` batch: same
license rules, same drop rule, same clip format, plus a queue, chunking, a face prescreen, N capture
workers, and rolling pushes so the whole thing runs on a machine with 18 GB of free disk.

Hub dataset: **`squaredcuber/animacy-human-motion-large`** (new repo; the curated 73-clip
`squaredcuber/animacy-human-motion` stays as is). Clips are directories
`clips/<shard>/<name>/{motion.parquet, audio.opus, meta.json}`; `index.json` is one row per clip;
`manifests/<shard>.json` per commit; the card carries the license policy and per-source counts.

## Pipeline

```
crawl.py   sources -> items(queued)          dedupe by id + title fingerprint, 60 s .. 60 min, title blocklist
fetch.py   queued -> fetched | refused       one paced download at a time, 480p <= 150 MB, license verified HERE for YouTube
workers.py fetched -> captured | dropped     N workers: 10-min chunks -> YuNet prescreen -> animacy capture -> drop rule -> clips/
push.py    clips(kept) -> pushed             one commit per ~50 kept hours (or when local clips > 2.5 GB), then delete locally
index.py   clips table -> _index.json        same shape as data/clips/_index.json + language/source/series tables
status.py  one screen                        hours by state, per-source yield, throughput 1/6/24 h, ETA, incidents, disk
daemon.py  supervisor                        runs the four loops, restarts on exit, STOP file to stop
```

The queue is `queue.sqlite` (tables `items`, `clips`, `events`, `kv`; WAL mode so 16 workers can
claim atomically). `index.py --dump-queue` writes the JSONL view (`queue.jsonl`). Item states:
`queued -> fetching -> fetched -> capturing -> captured | dropped`, plus `refused` (license),
`failed` (download/capture error), and per-chunk `prescreen` / `dropped: ...` records inside the
item. `manifest.jsonl` gets one line per kept clip at capture time and survives the local deletion
after push; `clips` rows do too.

### Sources and license evidence (mandatory, same refusal rules as `scripts/fetch_sources.py`)

| family | how it is listed | what proves the license | recorded in `license_evidence` |
|---|---|---|---|
| `ytsearch` | YouTube search with the Creative-Commons filter (`sp=EgIwAQ==`) under 3 sort orders, 154 queries in 29 languages (`sources.YT_QUERIES`) | `--match-filter "license ~= creative commons attribution"` before download, then the info-json `license` must classify as CC-BY (`fetch_sources.classify_license`) | `{"api": "yt-dlp info-json", "license": "Creative Commons Attribution license (reuse allowed)", channel_id, channel, upload_date, webpage_url}` |
| `usgov` | flat listing of 30 allowlisted official agency channels (`sources.USGOV_CHANNELS`: White House, State, DoD, NASA, Army/Navy/Air Force, CDC, USDA, NOAA, FEMA, USGS, NIST, 17 VOA services) | channel id of the fetched video == the allowlisted channel's id; basis 17 U.S.C. § 105 | `{"basis": "PD-USGov", "agency", "channel_url", "channel_id", "statement", ("caveat" for VOA)}` |
| `commons` | `data_fetch_more.plan_commons` over 17 categories (subcats 1 level) | `videoinfo.extmetadata` LicenseShortName / License / LicenseUrl / Copyrighted=False, re-read at fetch | as before |
| `archive` | `data_fetch_more.plan_archive` over 5 queries | item metadata `licenseurl` / `rights`, re-read at fetch | as before |

Refused by policy (not just by metadata): NIH VideoCast (visiting lecturers keep copyright),
party/caucus channels (not agencies), C-SPAN (private), anything ND / NC / SA / missing. VOA is
included with a caveat written into every clip: VOA's own productions are public domain, embedded
third-party newswire material is not; the single-face prescreen keeps studio segments and drops
package b-roll.

### Variety and caps

* No channel may exceed **5 %** of kept seconds (floor 3 h while the corpus is small). Enforced when
  `fetch.py` picks the next item (kept + 60 % of in-flight seconds per channel), so a big channel is
  passed over, not blocked, and re-opens as the total grows. `usgov` items get priority 0.5,
  `commons` 0.5, `archive` 0.3, `ytsearch` 0, each + uniform(0, 1) jitter, so sources interleave
  with a mild tilt toward the verified-at-crawl families.
* Per crawl round a YouTube channel adds at most 20 new search hits.
* Speaker key: YouTube = the channel (`yt:<channel_id>`, over-merges interview channels, which is the
  safe direction); Commons/archive = `data_capture_batch.speaker_key`. Series from title keywords.
* Language guess (recorded with its evidence): info-json `language`, else the `*-orig` automatic
  caption track (YouTube marks the spoken language that way), else the title's script, else the
  language the query was issued in.

### Chunking, prescreen, drop rule

* Items longer than 600 s are cut into equal stream-copied chunks of <= 600 s (`<slug>__c01` ...;
  a 25-min video is 3 x 500 s). `-ss` before `-i` starts each chunk at the keyframe at or before the
  cut, so a chunk may overlap the previous one by <= one GOP (~5 s); a re-encode would have cost
  ~10 % of the capture budget.
* Prescreen per chunk: 16 evenly sampled frames through YuNet (the same detector capture's
  small-face fallback uses); >= 60 % must show **one dominant face** (a second face narrower than
  half the first — a poster, a thumbnail — does not count), else the chunk is skipped
  without spending capture time. This is what turns a CC search full of screencasts and gameplay
  into an acceptable yield.
* Capture: `python -m animacy.cli capture --source <chunk> -o work/<slug>/<clip> --duration 600
  --neutral-seconds 0` (unchanged capture code). Drop rule as in `data_capture_batch.py index`:
  `face_valid >= 60 %` and `face_valid * duration >= 60 s`, license present.
* Kept clips: `audio.wav` -> `audio.opus` (32 kb/s mono; `ffmpeg -i audio.opus -ar 16000 -ac 1
  audio.wav` restores what `HumanClip.load` reads), `meta.json` gains `harvest` (item, channel,
  speaker key, language + evidence, chunk offset/length, prescreen numbers, worker) and `audio`
  blocks. Everything else (raw video, chunks, dropped captures) is deleted as soon as the item is done.

Why opus locally, not just in the pushed dataset: squaredcube1 has one 3.7 TB volume with **18 GB
free**; wav is 115 MB per kept hour (5,000 h = 575 GB), opus is 14 MB (72 GB). With opus the local
working set is the fetch buffer (<= 36 items, ~3 GB) plus at most ~2.5 GB of unpushed clips. Fetch
pauses below 6 GB free, capture below 3 GB; both incidents go to the events table.

## Hosts

**squaredcube1 (Windows native, `C:\harvest`)** — chosen over WSL because the host exposes 64
threads (WSL is capped at 24 in `.wslconfig`) and because WSL's ext4 vhdx grows on the same 18-GB
volume and never shrinks. GPUs untouched (`CUDA_VISIBLE_DEVICES=""`, torch CPU wheel).

```
C:\harvest\animacy       clone of the public repo (scripts/harvest/ is copied in by scp until it is committed)
C:\harvest\venv          python 3.12.13 (uv-managed interpreter) + pip install -e ".[capture]" yt-dlp[default] silero-vad huggingface_hub
C:\harvest\bin           ffmpeg/ffprobe (BtbN static gpl build), deno (yt-dlp >= 2025.11 needs a JS runtime for YouTube)
C:\harvest\data          HARVEST_ROOT (queue.sqlite, raw/, work/, clips/, manifest.jsonl, logs/, _index.json)
C:\harvest\hv.ps1        run any script in the venv with the environment set:  powershell -File C:\harvest\hv.ps1 scripts/harvest/status.py
```

Provisioning: `scripts/harvest/setup_windows.ps1` (idempotent; note the pip PATH fix for the Windows
"untrusted mount point" junction error). Start: `powershell -File C:\harvest\start_windows.ps1` (creates the daemon through WMI
`Win32_Process.Create`; a `Start-Process` child dies with the OpenSSH session, a WMI-created one does
not). Stop: create `C:\harvest\data\STOP` or kill the daemon; `daemon.py` restarts any child that
exits (60 s delay).

**Rented Linux box (later)** — `scripts/harvest/worker.sh`: clone, CPU torch, venv, static ffmpeg,
deno, then `daemon.py --n $HARVEST_N`. Set `HF_TOKEN` for pushes and `HARVEST_HF_REPO` if not the
default. Several boxes can push to the same repo: shard numbers are per box (`kv.next_shard`), so
give each box a distinct `HARVEST_SHARD_PREFIX`-style start by seeding `next_shard` (e.g. 1000,
2000) before the first push, and dedupe across boxes by giving each a disjoint slice of
`YT_QUERIES` (`crawl.py` accepts nothing for that yet — split `sources.py` per box).

## Throughput and ETA

Measured on squaredcube1 (fill in from `status.py` as the run progresses; first numbers below are the
smoke test): see the "Measured" section at the end of this file.

Per-worker capture speed sets everything: MediaPipe FaceLandmarker + PoseLandmarker-lite on every
frame at 480p is ~1x realtime per process on this CPU. With W workers and a survival fraction s
(kept seconds / captured seconds after the prescreen), kept hours per wall hour ~ W * speed * s.
The prescreen keeps capture time from being spent on no-face chunks, so s is the *post-prescreen*
survival (~0.6-0.8), and the prescreen rejection rate is reported separately.

Capture-side speed lever not taken (needs a capture flag, capture agent owns `animacy/capture.py`):
**`--pose-every N`** — run PoseLandmarker on every N-th processed frame and hold/interpolate torso
and arm samples between (the face path stays per-frame; `resample_to_grid` already interpolates
between valid neighbours, so pose samples every 2nd frame would just be a sparser `t_src` for the
torso/arm groups). Expected ~30-40 % less CPU per clip (pose-lite is roughly as expensive as the
face landmarker on 480p). `mirror` already has `--pose-every`; `capture` does not.

## Milestones and reporting

* (a) ~2 h: provisioned, 16 workers up, measured throughput and an honest ETA;
* (b) continuous; status line to the lead every ~2 h or every +100 kept hours;
* (c) push every ~50 kept hours (`push.py --loop --min-hours 50`, also pushes early when local clips
  exceed 2.5 GB).

`status.py` prints: kept/pushed/local hours, queue by state, throughput 1/6/24 h, kept-hours-per-
wall-hour and ETA to 5,000 h, per-source yield (kept / captured / fetched hours, chunks kept, refused,
failed), chunks kept vs dropped-after-capture vs prescreen-rejected in the last 24 h, hours by
language/series, top channels, rate-limit incidents in 24 h, disk, worker heartbeat.

## Measured

(filled in as the run progresses)
