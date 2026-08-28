#!/usr/bin/env bash
# Boot Autonomous OS HAL in its own laptop-simulator mode (the `make sim` path:
# HAL_SIMULATE=1 HAL_BOARD=sim, mock motion driver hal/drivers/motors/mock_service.py)
# from an UNMODIFIED scratch clone. Linux / WSL only: hal/ imports the Unix-only
# stdlib module `pwd` (hal/drivers/bluetooth_manager.py) on every body, so it
# cannot boot on native Windows. Nothing from the clone is vendored (hal/ is GPL-3.0).
#
#   bash scripts/lamp_hal_sim_start.sh --check      # import HAL, print the mounted routes, exit
#   bash scripts/lamp_hal_sim_start.sh              # serve on 0.0.0.0:5001 (foreground)
#   DEVICE_TYPE=lamp bash scripts/lamp_hal_sim_start.sh   # the Lamp body (needs their full driver stack)
#
# Env: AOS_REPO (clone path), AOS_VENV (python venv, created on first run with uv),
#      DEVICE_TYPE (sim|lamp, default sim), HAL_PORT (5001), SIM_STATE_DIR, HAL_LOG_DIR.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${AOS_REPO:-$HERE/../third_party/autonomous-os}"
VENV="${AOS_VENV:-$HOME/hal-aos-venv}"
export DEVICE_TYPE="${DEVICE_TYPE:-sim}"
PORT="${HAL_PORT:-5001}"
STATE="${SIM_STATE_DIR:-/tmp/autonomous-sim}"
LOGS="${HAL_LOG_DIR:-/tmp/hal-aos-logs}"
mkdir -p "$STATE" "$LOGS"

export PATH="$HOME/.local/bin:$PATH"   # uv's default install location (non-login shells miss it)
# The minimal subset of hal/pyproject.toml that the sim body needs; the full
# `uv sync` pulls lerobot/torch/ultralytics/livekit which the mock body never imports.
DEPS="fastapi uvicorn python-dotenv python-multipart pydantic numpy requests"
if [ ! -x "$VENV/bin/python" ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 "$VENV"
    uv pip install --python "$VENV/bin/python" $DEPS
  else
    python3.12 -m venv "$VENV"
    "$VENV/bin/pip" install $DEPS
  fi
fi

cd "$REPO/hal"
echo "autonomous-os commit: $(git -c safe.directory='*' -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
export PYTHONPATH=.. HAL_MODE=developer HAL_SIMULATE=1 HAL_SIM_MEDIA=virtual HAL_BOARD=sim
export HAL_LOG_DIR="$LOGS" HAL_USERS_DIR="$STATE/users" HAL_STRANGERS_DIR="$STATE/strangers"
export HAL_BT_STATE_DIR="$STATE" HAL_VOLUME_STATE_PATH="$STATE/volume"

if [ "${1:-}" = "--check" ]; then
  exec "$VENV/bin/python" -c 'import hal.server as s; print("IMPORT OK device=%s mounted=%s skipped=%s" % (s._resolve_device_type(), sorted(s._plan.mounted), sorted(s._plan.skipped)))'
fi
exec "$VENV/bin/uvicorn" hal.server:app --host 0.0.0.0 --port "$PORT"
