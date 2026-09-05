#!/bin/bash
set -e

# Optional: Use a virtual environment
VENV_DIR="ADsGUI"

echo "🔍 Checking for Python..."
if ! command -v python3 &>/dev/null; then
    echo "❌ Python3 is not installed. Please install it first."
    exit 1
fi

# Create a virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

echo "✅ Virtual environment activated."

# Upgrade pip
pip install --upgrade pip

# Install the specred package itself (editable) plus its GUI extra.
# This replaces the old per-package REQUIRED_PACKAGES loop: dependencies
# now live in pyproject.toml (single source of truth, no risk of the
# install script and the code silently drifting apart again -- the old
# script was missing PyYAML and scikit-learn, both of which the code
# actually imports).
#
# Run this from the repo root (where pyproject.toml lives).
echo "📦 Installing specred (editable) + GUI dependencies..."
pip install -e ".[gui]"

echo "✅ All dependencies are installed."
echo ""
echo "Run the GUI with:"
echo "    source $VENV_DIR/bin/activate"
echo "    specred-gui"
echo "or, without the console script:"
echo "    python -m gui.main_window"
