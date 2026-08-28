# Large-scale harvest (`scripts/harvest/`)

Goal: >= 5,000 hours of license-clean, single-speaker talking-head human motion in the canonical
clip format (`docs/CANONICAL.md`), harvested continuously and pushed to the Hub in shards. This is
the industrial version of last night's `data_fetch_more.py` / `data_capture_batch.py` batch: same
license rules, same drop rule, same clip format, plus a queue, chunking, a face prescreen, N capture
workers, and rolling pushes so the whole thing runs on a machine with 18 GB of free disk.

Context: [`CANONICAL.md`](CANONICAL.md) defines the clip format this produces,
[`MODEL.md`](MODEL.md) is what consumes it, and [`RESULTS.md`](RESULTS.md)
records what each corpus size actually bought. Docs index: [`README.md`](README.md).
The curated corpus that today's shipped numbers come from is **73 clips /
320.8 valid minutes / 37 speakers**; this pipeline is the path past that.

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

**Rented Linux box (Vast etc., 32-64 cores, Ubuntu, no GPU) — joining the same harvest**

No shared queue is needed. Every box crawls the same sources, but a box only *fetches* the items
whose id hashes into its slice (`HARVEST_PART="k/N"`, `common.item_part`: sha1(id) mod 1000, bucket
mod N == k), names its shards with its own prefix (`HARVEST_BOX`, e.g. `sb0001`), and `push.py`
rebuilds `index.json` from the union of every `manifests/*.json` on the Hub, so boxes never
overwrite each other. The 5 % channel cap is per box, which keeps the global share <= 5 % too.

Exact steps. Tested 2026-08-27 in WSL Ubuntu 24.04 on squaredcube1 (`HARVEST_N=2`, partition 1/2, box
`t`, scratch repo): clone + venv + deno + CPU torch in ~7 min, 19 clips / 2.56 h kept in 0.8 h from two
workers, shard `st0001` pushed from Linux with the union index; the scratch repo and directory were
deleted afterwards.

```
# 1. on squaredcube1: shrink its slice so the new box gets the rest (edit start_windows.ps1 / hv.ps1
#    env: HARVEST_PART=0/2 HARVEST_BOX="" ; then restart fetch: kill fetch.py, the daemon respawns it)
# 2. on the new box (as any user with python3 >= 3.10, git, curl):
curl -fsSL https://raw.githubusercontent.com/Hcoder10/animacy/master/scripts/harvest/worker.sh -o worker.sh
HARVEST_PART=1/2 HARVEST_BOX=b HF_TOKEN=hf_xxx HARVEST_N=32 nohup bash worker.sh > worker.out 2>&1 &
#    (until scripts/harvest is committed: scp -r scripts/harvest box:/tmp/harvest and add
#     HARVEST_SCRIPTS_SRC=/tmp/harvest to the line above)
# 3. watch:  ~/animacy-harvest/venv/bin/python ~/animacy-harvest/animacy/scripts/harvest/status.py
#    logs in ~/animacy-harvest/data/logs/, stop with: touch ~/animacy-harvest/data/STOP
```

`worker.sh` does: clone (or pull) the repo, static ffmpeg + deno into `~/animacy-harvest/bin`, venv
with CPU torch (from the PyTorch CPU index, so silero-vad does not pull the CUDA wheels), `pip
install -e ".[capture]" yt-dlp[default] silero-vad huggingface_hub soundfile`, prints the partition it
will serve, then `exec daemon.py --n $HARVEST_N`. `HARVEST_N` defaults to nproc/2; each worker is
~1.5x realtime, so a 64-core box at N=32 captures ~45 h of video per hour. Disk: the box needs
~4 GB for the venv + fetch buffer (36 items) + < 3 GB of unpushed clips; nothing accumulates because
pushed clips are deleted. A third box is `HARVEST_PART=2/3` with the others re-sliced to 0/3 and 1/3.
Change N on every box at the same time (kill fetch.py on each; the daemons respawn it): the slices
are a pure function of the item id, so the only duplicates possible are items that were already in
flight on a box whose slice they just left.

## Throughput and ETA

Measured on squaredcube1 (fill in from `status.py` as the run progresses; first numbers below are the
smoke test): see the "Measured" section at the end of this file.

Per-worker capture speed sets everything: MediaPipe FaceLandmarker + PoseLandmarker-lite on every
frame at 480p is ~1x realtime per process on this CPU. With W workers and a survival fraction s
(kept seconds / captured seconds after the prescreen), kept hours per wall hour ~ W * speed * s.
The prescreen keeps capture time from being spent on no-face chunks, so s is the *post-prescreen*
survival (~0.6-0.8), and the prescreen rejection rate is reported separately.

Capture-side speed lever: **`--pose-every 2`** landed in e9f04cc (capture agent; ~20 % less CPU,
31 % less wall) together with a frame-decimation fix (29.97 fps sources were sampled at ~20 fps and
grid-interpolated; now true 30/s). Workers pass `--pose-every $HARVEST_POSE_EVERY` (default 2) when
the checkout supports it, and every clip records `capture_git` + `pose_every` in
`meta.json.harvest` and in the index rows. Clips captured before e9f04cc (no `capture_git` field,
`s0001` and the first ~150 clips) are kept: they are 20 -> 30 fps interpolated, not wrong, and the
field tells them apart. Code reloads without killing in-flight captures: `touch <ROOT>/RELOAD`
(`workers.py`) — or `reload_workers.py` for an instance that predates the hook.

## Milestones and reporting

* (a) ~2 h: provisioned, 16 workers up, measured throughput and an honest ETA;
* (b) continuous; status line to the lead every ~2 h or every +100 kept hours;
* (c) push every ~50 kept hours (`push.py --loop --min-hours 50`, also pushes early when local clips
  exceed 2.5 GB).

`status.py` prints: kept/pushed/local hours, queue by state, throughput 1/6/24 h, kept-hours-per-
wall-hour and ETA to 5,000 h, per-source yield (kept / captured / fetched hours, chunks kept, refused,
failed), chunks kept vs dropped-after-capture vs prescreen-rejected in the last 24 h, hours by
language/series, top channels, rate-limit incidents in 24 h, disk, worker heartbeat.

## Incidents

* **2026-08-27 15:19 — YouTube "Sign in to confirm you're not a bot"** on squaredcube1 after ~1.5 h
  at ~300 downloads/h (8 s pacing) plus the WSL test crawling/fetching from the same host. Diagnosis:
  the block is per address and the Windows host was going out over **IPv6**; a `-4` probe passed
  while `-6` failed, and WSL (IPv4 NAT) kept fetching throughout. Fixes: every yt-dlp call forces
  IPv4 (`common.YTDLP_COMMON`), pacing 20 s mean (~150/h), YouTube-only back-off (5/10/20/30 min,
  `--simulate` probe before resuming) while Commons/archive.org keep flowing, and the one-off
  Commons + archive crawl was run so the queue has non-YouTube supply. Cookies were not and will not
  be used (the owner's account). Lost: ~1 h of fetch (workers idled once the buffer drained).

* **2026-08-27 16:17 — Commons crawl HTTP 414** (50 long interview titles per `videoinfo` GET):
  crawl now batches 12 titles and survives a failing category. **archive.org advancedsearch**
  returned `[BACKEND_ERROR] Invalid or no response from Elasticsearch` for every query (their
  side); the crawl loop retries each round, no archive items in the queue until it recovers.
* **2026-08-27 16:2x — YouTube CC label**: yt-dlp's license string carries no version, so
  `classify_license` labelled YouTube clips CC-BY-4.0; YouTube's CC option is CC BY 3.0.
  `fetch.py` now labels `CC-BY-3.0`; `fix_cc_label.py` rewrote the ~110 affected rows / meta.json
  files before they were pushed (shard s0001 was gov-only, nothing on the Hub was wrong).

## Measured

**2026-08-27 14:15 PDT, launch on squaredcube1, 16 workers (first 12 min).**

* Per-worker capture speed with 16 concurrent processes: **1.44-2.04x realtime** at 854x480
  (e.g. 8.3 min chunk in 245 s wall, 4.9 min in 201 s). Host CPU ~57 % (64 threads), so 16 is not
  the ceiling; ~24 workers should fit without touching the GPU job.
* Fetch: ~5 s per 480p download (8-110 MB, AVC), 12 s mean pacing -> ~20 h of video per hour from
  YouTube with 0 rate-limit incidents in the first 300 downloads.
* Prescreen cost ~1-2 s per chunk (16 seeks + YuNet). On the White House / State Dept feeds it
  rejected 32 chunks and passed 6 (event and crowd footage vs. remarks-to-camera); the 6 passed
  chunks all survived the drop rule (face_valid 0.9+), i.e. the prescreen predicts the gate well.
* Early gov-only survival: 15 % of fetched source time kept. CC-search vlog/interview content is
  expected much higher (numbers to follow as the mix shifts; 2,005 items / 654 h queued at launch).
* Capture capacity therefore ~25-30 h/h; kept hours per wall hour = fetch rate x survival, so the
  levers are (1) survival via source choice, (2) fetch pacing (YouTube tolerance), (3) more boxes.
