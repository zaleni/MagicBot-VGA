#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="${CONDA_ENV:-}"
LIBERO_HOME="${LIBERO_HOME:-$HOME/LIBERO}"
LIBERO_GIT_URL="${LIBERO_GIT_URL:-https://github.com/Lifelong-Robot-Learning/LIBERO.git}"
INSTALL_EXTRA_EVAL_DEPS="${INSTALL_EXTRA_EVAL_DEPS:-true}"
CHECK_IMPORTS="${CHECK_IMPORTS:-true}"

if [[ -n "${CONDA_ENV}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "CONDA_ENV is set but 'conda' was not found in PATH."
    exit 1
  fi
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "PROJ_ROOT=${PROJ_ROOT}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "LIBERO_HOME=${LIBERO_HOME}"

if ! command -v git >/dev/null 2>&1; then
  echo "'git' is required to clone LIBERO."
  exit 1
fi

echo "Installing base evaluation dependencies..."
"${PYTHON_BIN}" -m pip install mujoco tyro imageio imageio-ffmpeg

if [[ "${INSTALL_EXTRA_EVAL_DEPS}" == "true" ]]; then
  echo "Installing optional helper dependencies..."
  "${PYTHON_BIN}" -m pip install matplotlib mediapy websockets msgpack
fi

if [[ ! -d "${LIBERO_HOME}" ]]; then
  echo "Cloning LIBERO into ${LIBERO_HOME}..."
  git clone "${LIBERO_GIT_URL}" "${LIBERO_HOME}"
fi

if [[ ! -f "${LIBERO_HOME}/setup.py" && ! -f "${LIBERO_HOME}/pyproject.toml" ]]; then
  echo "LIBERO_HOME does not look like a valid LIBERO checkout: ${LIBERO_HOME}"
  exit 1
fi

echo "Installing LIBERO in editable mode..."
(
  cd "${LIBERO_HOME}"
  "${PYTHON_BIN}" -m pip install -e .
)

if [[ "${CHECK_IMPORTS}" == "true" ]]; then
  echo "Verifying installation..."
  "${PYTHON_BIN}" - <<'PY'
import imageio
import mujoco
import tyro
from libero.libero import benchmark

print("LIBERO OK:", benchmark is not None)
print("MuJoCo OK:", mujoco.__version__)
print("imageio OK:", imageio.__version__)
print("tyro OK:", tyro.__version__)
PY
fi

cat <<'EOF'

Installation finished.

If you are on a headless server, these environment variables are commonly needed before evaluation:

  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl

You can then run:

  bash evaluation/Libero/eval.sh

EOF
