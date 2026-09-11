#!/usr/bin/env bash
# Sets up (on first run) and launches the video pipeline. Linux/macOS.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

VENV_DIR="$PROJECT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Installing/upgrading dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo
python3 scripts/inspect_env.py || {
    echo
    echo "Environment check failed (see above). Fix ffmpeg/Python and re-run."
    exit 1
}

if [ ! -f "config.yaml" ]; then
    echo
    echo "No config.yaml found — copying config.example.yaml -> config.yaml"
    cp config.example.yaml config.yaml
fi

echo
echo "Starting video pipeline. Press Ctrl+C to stop."
echo
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
exec python3 -m video_pipeline.main --config config.yaml
