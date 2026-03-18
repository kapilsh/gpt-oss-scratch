#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup.sh — create a uv virtual environment for gpt-oss-scratch
#
# Usage:
#   ./setup.sh          # create .venv and install everything
#   ./setup.sh --help
#
# Requirements:
#   - uv  (https://docs.astral.sh/uv/getting-started/installation/)
#   - CUDA 13.0 drivers
# ---------------------------------------------------------------------------

set -euo pipefail

VENV_DIR=".venv"
PYTHON_VERSION="3.12"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
DIM='\033[2m'
RESET='\033[0m'

step() { echo -e "\n${CYAN}▶ $*${RESET}"; }
ok()   { echo -e "${GREEN}✓ $*${RESET}"; }
die()  { echo -e "${RED}✗ $*${RESET}" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Check prerequisites
# ---------------------------------------------------------------------------
step "Checking prerequisites"

command -v uv >/dev/null 2>&1 || die "uv not found. Install it from https://docs.astral.sh/uv/getting-started/installation/"
ok "uv $(uv --version)"

# Ensure the gpt-oss submodule is initialised
if [ ! -f "gpt-oss/pyproject.toml" ]; then
    step "Initialising git submodules"
    git submodule update --init --recursive
    ok "Submodules ready"
fi

# ---------------------------------------------------------------------------
# Create virtual environment
# ---------------------------------------------------------------------------
step "Creating virtual environment (Python ${PYTHON_VERSION})"
uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}"
ok "Created ${VENV_DIR}/"

# Activate for the remainder of this script
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# ---------------------------------------------------------------------------
# PyTorch (CUDA 13.0 build — must come first so vllm picks up the right torch)
# ---------------------------------------------------------------------------
step "Installing PyTorch with CUDA 13.0"
uv pip install \
    "torch==2.10.0+cu130" \
    "torchvision==0.25.0+cu130" \
    "triton==3.6.0" \
    --extra-index-url "${TORCH_INDEX}"
ok "PyTorch installed"

# ---------------------------------------------------------------------------
# gpt-oss submodule (editable, with triton extras)
# ---------------------------------------------------------------------------
step "Installing gpt-oss submodule [triton]"
uv pip install -e "gpt-oss[triton]"
ok "gpt-oss installed"

# ---------------------------------------------------------------------------
# Core project dependencies
# ---------------------------------------------------------------------------
step "Installing core dependencies"
uv pip install \
    accelerate \
    transformers \
    safetensors \
    huggingface-hub \
    tiktoken \
    openai-harmony \
    fastapi \
    uvicorn \
    pydantic \
    click \
    structlog \
    rich \
    prometheus-client \
    pandas \
    numpy \
    tensorboard \
    torch-tb-profiler \
    jupyter \
    ipykernel \
    ipywidgets
ok "Core dependencies installed"

# ---------------------------------------------------------------------------
# vLLM
# ---------------------------------------------------------------------------
step "Installing vLLM"
uv pip install vllm==0.17.1
ok "vLLM installed"

# ---------------------------------------------------------------------------
# Register the kernel for Jupyter
# ---------------------------------------------------------------------------
step "Registering Jupyter kernel"
python -m ipykernel install --user --name gpt-oss --display-name "gpt-oss (Python ${PYTHON_VERSION})"
ok "Kernel registered"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${GREEN}  Environment ready!${RESET}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${DIM}"
echo "  Activate:    source ${VENV_DIR}/bin/activate"
echo "  Serve model: python gpt_oss_vllm_serve.py --tp 2"
echo "  Manual REPL: python gpt_oss_manual_inference.py"
echo "  Jupyter:     jupyter lab"
echo -e "${RESET}"
