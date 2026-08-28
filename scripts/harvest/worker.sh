#!/usr/bin/env bash
# Harvest worker for a rented Linux box (CPU-only, Ubuntu 22.04/24.04, python3 >= 3.10).
#
#   HARVEST_PART=1/2 HARVEST_BOX=b HF_TOKEN=hf_... HARVEST_N=32 bash worker.sh
#
# Env:
#   HARVEST_PART   "k/N" — this box fetches only items whose id hashes into slice k of N
#                  (squaredcube1 runs 0/2 when one box joins, 0/3 with two, ...). Default 0/1.
#   HARVEST_BOX    one-letter shard prefix, unique per box (shards become s<box>0001). Default "".
#   HF_TOKEN       Hugging Face write token for the dataset repo (pushes are refused without it).
#   HARVEST_N      capture workers, default nproc/2 (each ~1.5x realtime; leave cores for ffmpeg + yt-dlp).
#   HARVEST_HF_REPO  dataset repo, default squaredcuber/animacy-human-motion-large.
#   HARVEST_BASE   install dir, default ~/animacy-harvest (data in $HARVEST_BASE/data).
#   HARVEST_SCRIPTS_SRC  optional: a scripts/harvest directory to copy over the clone's (until the
#                  harvest scripts are committed to the public repo).
set -euo pipefail
BASE="${HARVEST_BASE:-$HOME/animacy-harvest}"
export HARVEST_ROOT="${HARVEST_ROOT:-$BASE/data}"
export HARVEST_BIN="${HARVEST_BIN:-$BASE/bin}"
export HARVEST_PART="${HARVEST_PART:-0/1}"
export HARVEST_BOX="${HARVEST_BOX:-}"
export CUDA_VISIBLE_DEVICES=""
mkdir -p "$BASE" "$HARVEST_ROOT" "$HARVEST_BIN"
if [ ! -d "$BASE/animacy/.git" ]; then git clone https://github.com/Hcoder10/animacy.git "$BASE/animacy"; else git -C "$BASE/animacy" pull --ff-only || true; fi
if [ -n "${HARVEST_SCRIPTS_SRC:-}" ]; then mkdir -p "$BASE/animacy/scripts/harvest" && cp "$HARVEST_SCRIPTS_SRC"/*.py "$HARVEST_SCRIPTS_SRC"/worker.sh "$BASE/animacy/scripts/harvest/"; fi
if [ ! -f "$BASE/animacy/scripts/harvest/daemon.py" ]; then
  echo "scripts/harvest/ is not in the clone (not committed yet): set HARVEST_SCRIPTS_SRC=<path to scripts/harvest> or scp it into $BASE/animacy/scripts/harvest/" >&2; exit 1
fi
if ! command -v ffmpeg >/dev/null && [ ! -x "$HARVEST_BIN/ffmpeg" ]; then
  curl -fsSL -o /tmp/ffmpeg.tar.xz https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz
  tar -xJf /tmp/ffmpeg.tar.xz -C /tmp && cp /tmp/ffmpeg-master-latest-linux64-gpl/bin/ff* "$HARVEST_BIN/" && rm -rf /tmp/ffmpeg.tar.xz /tmp/ffmpeg-master-latest-linux64-gpl
fi
if ! command -v deno >/dev/null && [ ! -x "$HARVEST_BIN/deno" ]; then
  curl -fsSL -o /tmp/deno.zip https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip
  (cd "$HARVEST_BIN" && python3 -c "import zipfile; zipfile.ZipFile('/tmp/deno.zip').extractall('.')" && chmod +x deno) && rm -f /tmp/deno.zip
fi
if [ ! -x "$BASE/venv/bin/python" ]; then python3 -m venv "$BASE/venv"; fi
"$BASE/venv/bin/pip" install -q -U pip
# CPU torch first so silero-vad does not drag the CUDA wheels in
"$BASE/venv/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
"$BASE/venv/bin/pip" install -q -e "$BASE/animacy[capture]" "yt-dlp[default]" silero-vad huggingface_hub soundfile
export PATH="$HARVEST_BIN:$PATH"
if [ -z "${HF_TOKEN:-}" ] && [ ! -f "$HOME/.cache/huggingface/token" ]; then echo "WARNING: no HF_TOKEN; pushes will fail (capture still runs)" >&2; fi
N="${HARVEST_N:-$(( $(nproc) / 2 ))}"
cd "$BASE/animacy"
"$BASE/venv/bin/python" -c "import sys; sys.path.insert(0,'scripts/harvest'); import common as C; print('harvest root', C.ROOT, 'partition', C.PART_K, '/', C.PART_N, 'box', repr(C.BOX), 'ffmpeg', C.bin_path('ffmpeg'))"
exec "$BASE/venv/bin/python" scripts/harvest/daemon.py --n "$N" ${HARVEST_HF_REPO:+--repo "$HARVEST_HF_REPO"} "$@"
