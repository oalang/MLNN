#!/usr/bin/env bash
set -euo pipefail

# Config
PYTHON_BIN="${PYTHON_BIN:-python3.13}"   # override if needed, e.g. PYTHON_BIN=python3.12
VENV_DIR=".venv-mlnn"
KERNEL_NAME="mlnn"
KERNEL_DISPLAY_NAME="Python (mlnn)"

# Create venv with latest stable Python (3.13 by default)
$PYTHON_BIN -m venv "$VENV_DIR"

# Activate venv
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Upgrade build tooling
python -m pip install --upgrade pip setuptools wheel

# Core scientific stack (aligned with JAX policy: NumPy >= 2.0, SciPy >= 1.13)
python -m pip install \
  "numpy>=2.0" \
  "scipy>=1.13" \
  "scikit-learn>=1.5" \
  matplotlib

# JAX (CPU wheel). For GPU, see: https://jax.readthedocs.io/en/latest/installation.html
python -m pip install --upgrade "jax[cpu]"

# Jupyter kernel + UI (+ interactive matplotlib backend)
python -m pip install ipykernel jupyterlab ipympl

# Install this project
python -m pip install -e .

# Register Jupyter kernel
python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME"

echo ""
echo "Environment ready."
echo "Venv: $(realpath "$VENV_DIR")"
echo "Kernel: $KERNEL_NAME"
echo ""
echo "Use this environment consistently:"
echo "- Terminal/Chat: source \"$VENV_DIR/bin/activate\""
echo "- VS Code/Cursor: Python: Select Interpreter → $(realpath "$VENV_DIR")/bin/python"
echo "  Or .vscode/settings.json:"
echo "    { \"python.defaultInterpreterPath\": \"$(realpath "$VENV_DIR")/bin/python\", \"python.terminal.activateEnvironment\": true }"
echo "- Jupyter: pick kernel \"$KERNEL_DISPLAY_NAME\""