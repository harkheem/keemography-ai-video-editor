#!/usr/bin/env bash
# start.sh — starts the FastAPI backend and the React frontend dev server
# Usage: ./start.sh
# Stop with Ctrl+C

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Choose Python interpreter ────────────────────────────────────────────────
PYTHON="${PYTHON:-python3}"

# Try venv311 first (existing project venv), then venv, then system python3
if [ -f "venv311/bin/python" ]; then
  PYTHON="$SCRIPT_DIR/venv311/bin/python"
elif [ -f ".venv/bin/python" ]; then
  PYTHON="$SCRIPT_DIR/.venv/bin/python"
fi

echo "Using Python: $PYTHON"

# ── Install backend extra deps (fastapi, uvicorn) if not present ─────────────
"$PYTHON" -m pip install -q fastapi uvicorn[standard] python-multipart 2>&1 | tail -5

# ── Install frontend deps if node_modules is missing ────────────────────────
if [ ! -d "frontend/node_modules" ]; then
  echo "Installing frontend dependencies..."
  NPM=$(command -v npm 2>/dev/null || echo "/opt/homebrew/bin/npm")
  cd frontend && "$NPM" install && cd ..
fi

NPM=$(command -v npm 2>/dev/null || echo "/opt/homebrew/bin/npm")
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Starting Keemography"
echo "  Backend  → http://localhost:8000"
echo "  Frontend → http://localhost:5173"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run backend in background
"$PYTHON" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Run frontend dev server
cd frontend && "$NPM" run dev &
FRONTEND_PID=$!

# Cleanup on Ctrl+C
cleanup() {
  echo ""
  echo "Shutting down…"
  kill $BACKEND_PID 2>/dev/null || true
  kill $FRONTEND_PID 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

wait
