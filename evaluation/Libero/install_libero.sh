#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="${CONDA_ENV:-}"
LIBERO_HOME_DEFAULT="${PROJ_ROOT}/third_party/LIBERO"
if [[ ! -d "${LIBERO_HOME_DEFAULT}" ]]; then
  LIBERO_HOME_DEFAULT="$HOME/LIBERO"
fi
LIBERO_HOME="${LIBERO_HOME:-${LIBERO_HOME_DEFAULT}}"
LIBERO_GIT_URL="${LIBERO_GIT_URL:-https://github.com/Lifelong-Robot-Learning/LIBERO.git}"
INSTALL_EXTRA_EVAL_DEPS="${INSTALL_EXTRA_EVAL_DEPS:-true}"
INSTALL_MINIMAL_LIBERO_EVAL_DEPS="${INSTALL_MINIMAL_LIBERO_EVAL_DEPS:-true}"
PIN_LIBERO_COMPAT_DEPS="${PIN_LIBERO_COMPAT_DEPS:-false}"
INSTALL_LIBERO_REQUIREMENTS="${INSTALL_LIBERO_REQUIREMENTS:-false}"
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

echo "Installing base evaluation dependencies..."
"${PYTHON_BIN}" -m pip install mujoco tyro imageio imageio-ffmpeg

if [[ "${INSTALL_EXTRA_EVAL_DEPS}" == "true" ]]; then
  echo "Installing optional helper dependencies..."
  "${PYTHON_BIN}" -m pip install matplotlib mediapy websockets msgpack
fi

if [[ ! -d "${LIBERO_HOME}" ]]; then
  if ! command -v git >/dev/null 2>&1; then
    echo "'git' is required to clone LIBERO."
    exit 1
  fi
  echo "Cloning LIBERO into ${LIBERO_HOME}..."
  git clone "${LIBERO_GIT_URL}" "${LIBERO_HOME}"
fi

if [[ ! -f "${LIBERO_HOME}/setup.py" && ! -f "${LIBERO_HOME}/pyproject.toml" ]]; then
  echo "LIBERO_HOME does not look like a valid LIBERO checkout: ${LIBERO_HOME}"
  exit 1
fi

if [[ "${INSTALL_MINIMAL_LIBERO_EVAL_DEPS}" == "true" ]]; then
  if [[ "${PIN_LIBERO_COMPAT_DEPS}" == "true" ]]; then
    echo "Installing minimal LIBERO eval dependencies with conservative compatibility pins..."
    "${PYTHON_BIN}" -m pip install pyyaml "bddl==1.0.1" "robosuite==1.4.0"
  else
    echo "Installing minimal LIBERO eval dependencies without forcing the full upstream requirements set..."
    "${PYTHON_BIN}" -m pip install pyyaml bddl robosuite
  fi
fi

if [[ "${INSTALL_LIBERO_REQUIREMENTS}" == "true" && -f "${LIBERO_HOME}/requirements.txt" ]]; then
  echo "Installing full LIBERO requirements from ${LIBERO_HOME}/requirements.txt..."
  echo "Warning: upstream LIBERO pins many package versions and may downgrade an existing MagicBot environment."
  "${PYTHON_BIN}" -m pip install -r "${LIBERO_HOME}/requirements.txt"
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
import bddl
from libero.libero import benchmark
import robosuite
from libero.libero.envs import OffScreenRenderEnv

print("LIBERO OK:", benchmark is not None)
print("OffScreenRenderEnv OK:", OffScreenRenderEnv is not None)
print("MuJoCo OK:", mujoco.__version__)
print("imageio OK:", imageio.__version__)
print("tyro OK:", tyro.__version__)
print("bddl OK:", getattr(bddl, "__version__", "unknown"))
print("robosuite OK:", getattr(robosuite, "__version__", "unknown"))
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
