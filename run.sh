#!/bin/bash
# Automatically use the virtual environment python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Virtual environment not found. Installing..."
    python3 -m venv .venv
    "$VENV_PYTHON" -m pip install -r requirements.txt
fi

"$VENV_PYTHON" "$SCRIPT_DIR/auto_scanlate.py" "$@"
