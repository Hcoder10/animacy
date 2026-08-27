#!/usr/bin/env bash
# One-line harvest worker for a rented Linux box (CPU-only):
#   curl -fsSL https://raw.githubusercontent.com/Hcoder10/animacy/master/scripts/harvest/worker.sh | HARVEST_N=32 bash
# or from a clone:  HARVEST_N=32 scripts/harvest/worker.sh
# Env: HARVEST_N (capture workers, default nproc/2), HARVEST_ROOT (data dir), HARVEST_HF_REPO, HF_TOKEN (for push).
set -euo pipefail
BASE="${HARVEST_BASE:-$HOME/animacy-harvest}"
export HARVEST_ROOT="${HARVEST_ROOT:-$BASE/data}"
export HARVEST_BIN="${HARVEST_BIN:-$BASE/bin}"
mkdir -p "$BASE" "$HARVEST_ROOT" "$HARVEST_BIN"
if [ ! -d "$BASE/animacy/.git" ]; then git clone https://github.com/Hcoder10/animacy.git "$BASE/animacy"; else git -C "$BASE/animacy" pull --ff-only || true; fi
if ! command -v ffmpeg >/dev/null; then
  curl -fsSL -o /tmp/ffmpeg.tar.xz https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz
  tar -xJf /tmp/ffmpeg.tar.xz -C /tmp && cp /tmp/ffmpeg-master-latest-linux64-gpl/bin/ff* "$HARVEST_BIN/"
fi
if ! command -v deno >/dev/null && [ ! -x "$HARVEST_BIN/deno" ]; then
  curl -fsSL https://deno.land/install.sh | DENO_INSTALL="$BASE/deno" sh >/dev/null && cp "$BASE/deno/bin/deno" "$HARVEST_BIN/"
fi
if [ ! -x "$BASE/venv/bin/python" ]; then python3 -m venv "$BASE/venv"; fi
"$BASE/venv/bin/pip" install -q -U pip
# CPU torch first so silero-vad does not drag the CUDA wheels in
"$BASE/venv/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
"$BASE/venv/bin/pip" install -q -e "$BASE/animacy[capture]" "yt-dlp[default]" silero-vad huggingface_hub soundfile
export PATH="$HARVEST_BIN:$PATH"
N="${HARVEST_N:-$(( $(nproc) / 2 ))}"
cd "$BASE/animacy"
exec "$BASE/venv/bin/python" scripts/harvest/daemon.py --n "$N" ${HARVEST_HF_REPO:+--repo "$HARVEST_HF_REPO"}
