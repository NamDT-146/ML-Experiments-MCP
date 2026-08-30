#!/usr/bin/env bash
# Install aiaun-mcp + official mcp SDK. Prefer existing conda; fall back to new env or uv.
# Usage (WSL): bash /mnt/h/Dev/Lab/AiAuN/scripts/install_wsl.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MARKER="$ROOT/.aiaun-python"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" ]]; then
  CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
fi

py_ok() {
  local py="$1"
  "$py" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null
}

install_into() {
  local py="$1"
  echo "Installing into: $py ($("$py" -c 'import sys; print(sys.version.split()[0])'))"
  if ! "$py" -m pip install -e "${ROOT}[dev]" --default-timeout="$PIP_DEFAULT_TIMEOUT"; then
    return 1
  fi
  if ! "$py" -c "import mcp; import aiaun_mcp; print('mcp ok')"; then
    return 1
  fi
  printf '%s\n' "$py" > "$MARKER"
  echo "Wrote $MARKER"
}

try_conda_env() {
  local name="$1"
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$name" 2>/dev/null || return 1
  local py
  py="$(command -v python)"
  py_ok "$py" || { echo "skip $name: need Python >=3.10"; return 1; }
  install_into "$py"
}

if [[ -f "$CONDA_SH" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  if try_conda_env ssemi; then
    exit 0
  fi
  echo "ssemi failed or Python too old; trying conda base..."
  if try_conda_env base; then
    exit 0
  fi
fi

if command -v python3.12 >/dev/null 2>&1 && py_ok python3.12; then
  install_into "$(command -v python3.12)" && exit 0
fi
if command -v python3.11 >/dev/null 2>&1 && py_ok python3.11; then
  install_into "$(command -v python3.11)" && exit 0
fi
if command -v python3.10 >/dev/null 2>&1 && py_ok python3.10; then
  install_into "$(command -v python3.10)" && exit 0
fi

if command -v uv >/dev/null 2>&1; then
  echo "Trying uv venv at $ROOT/.venv ..."
  cd "$ROOT"
  uv python install 3.11 || true
  uv venv --python 3.11 .venv
  uv pip install -e "${ROOT}[dev]"
  .venv/bin/python -c "import mcp; import aiaun_mcp"
  printf '%s\n' "$ROOT/.venv/bin/python" > "$MARKER"
  echo "Wrote $MARKER (uv)"
  exit 0
fi

if [[ -f "$CONDA_SH" ]]; then
  echo "Creating conda env aiaun (python=3.11)..."
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda create -y -n aiaun python=3.11 pip
  conda activate aiaun
  install_into "$(command -v python)"
  exit 0
fi

echo "ERROR: could not find Python >=3.10 or conda/uv to install mcp" >&2
exit 1
