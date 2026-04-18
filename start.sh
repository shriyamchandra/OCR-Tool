#!/usr/bin/env bash
# ──────────────────────────────────────────────
# OCR-Tool — Start Script
# Sets up a virtual environment, installs
# dependencies, and launches the Streamlit app.
# ──────────────────────────────────────────────
set -euo pipefail

# Resolve the project root (directory where this script lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="venv"
REQUIREMENTS="requirements.txt"
APP_ENTRY="app.py"

# ── Colors ──
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}ℹ  $*${NC}"; }
ok()    { echo -e "${GREEN}✅ $*${NC}"; }
warn()  { echo -e "${YELLOW}⚠️  $*${NC}"; }

# ── 1. Check Python ──
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "❌ Python not found. Please install Python 3.8+ and try again."
    exit 1
fi

PY_VERSION=$($PYTHON --version 2>&1)
info "Using $PY_VERSION"

# ── 2. Create virtual environment if needed ──
if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
    ok "Virtual environment created at ./$VENV_DIR"
else
    ok "Virtual environment already exists"
fi

# ── 3. Activate virtual environment ──
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
info "Activated virtual environment"

# ── 4. Upgrade pip (quietly) ──
pip install --upgrade pip --quiet

# ── 5. Install dependencies ──
if [ -f "$REQUIREMENTS" ]; then
    info "Installing dependencies from $REQUIREMENTS..."
    pip install -r "$REQUIREMENTS" --quiet
    ok "Dependencies installed"
else
    warn "$REQUIREMENTS not found — skipping dependency install"
fi

# ── 6. Kill any existing Streamlit process (clears in-memory cache) ──
# @st.cache_data stores results in-memory inside the running process.
# The only reliable way to clear it is to restart the process entirely.
info "Stopping any running Streamlit instances..."
pkill -9 -f "streamlit run" 2>/dev/null && ok "Stopped existing Streamlit process" || true
sleep 2  # Give the OS time to release the port after a force kill

# Clear Python bytecode cache so any code changes are always picked up
find "$SCRIPT_DIR" -type d -name "__pycache__" \
    -not -path "*/venv/*" \
    -exec rm -rf {} + 2>/dev/null || true
ok "Cleared __pycache__"

# ── 7. Skip Streamlit onboarding prompt ──
# Streamlit asks for an email on first run, blocking the server from starting.
# Writing an empty credentials.toml tells it we've already been greeted.
STREAMLIT_DIR="$HOME/.streamlit"
CREDS_FILE="$STREAMLIT_DIR/credentials.toml"
if [ ! -f "$CREDS_FILE" ]; then
    mkdir -p "$STREAMLIT_DIR"
    cat > "$CREDS_FILE" << 'EOF'
[general]
email = ""
EOF
    ok "Skipped Streamlit onboarding prompt"
fi

# ── 8. Launch Streamlit ──
echo ""
echo -e "${GREEN}🚀 Starting OCR-Tool...${NC}"
echo -e "${CYAN}   The app will open in your browser at http://localhost:8501${NC}"
echo ""

streamlit run "$APP_ENTRY" \
    --server.headless true \
    --browser.gatherUsageStats false
