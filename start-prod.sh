#!/usr/bin/env bash
# start-prod.sh — builds React and serves everything from FastAPI on port 8000
# Usage: ./start-prod.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
if [ -f "venv311/bin/python" ]; then PYTHON="$SCRIPT_DIR/venv311/bin/python"; fi
if [ -f ".venv/bin/python" ]; then PYTHON="$SCRIPT_DIR/.venv/bin/python"; fi

NPM=$(command -v npm 2>/dev/null || echo "/opt/homebrew/bin/npm")
echo "Building React frontend…"
cd frontend && "$NPM" install && "$NPM" run build && cd ..

echo "Starting FastAPI server (serves built frontend too)…"
echo "Open http://localhost:8000"

$PYTHON -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
