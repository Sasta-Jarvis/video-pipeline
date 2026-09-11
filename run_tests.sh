#!/usr/bin/env bash
# Runs the full test suite. Generates fresh sample media on first run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

export PYTHONPATH="$SCRIPT_DIR/src:${PYTHONPATH:-}"

echo "=== Environment check ==="
python3 scripts/inspect_env.py || true
echo

echo "=== Generating sample test media (espeak-ng + ffmpeg lavfi) ==="
python3 tests/generate_sample_media.py
echo

echo "=== Running test suite ==="
python3 -m unittest discover -s tests -p "test_*.py" -v
