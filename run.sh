#!/usr/bin/env bash
set -euo pipefail

# ================== Config por defecto ==================
ENV_NAME="${ENV_NAME:-TG}"

# 🔧 Python 3.6 tanto en PC como Jetson para compatibilidad completa
PYTHON_VER="${PYTHON_VER:-3.6}"

# Puedes apuntar a índices de PyTorch si quieres CUDA en PC:
# ej: TORCH_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"
TORCH_EXTRA_INDEX_URL="${TORCH_EXTRA_INDEX_URL:-}"

# ========================================================
# Detectores de plataforma
is_cmd() { command -v "$1" >/dev/null 2>&1; }
is_aarch64() { [[ "$(uname -m)" == "aarch64" ]]; }
is_jetson_board() {
  [[ -f /etc/nv_tegra_release ]] && return 0
  [[ -d /proc/device-tree/tegra-fuse || -d /proc/device-tree/chosen/nvidia,tegra-udrm ]] && return 0
  return 1
}

IS_JETSON=0
if is_aarch64 && is_jetson_board; then
  IS_JETSON=1
fi

# ================== Pausa inteligente ===================
pause_if_needed() {
  local want_pause=0
  if [[ "${KEEP_OPEN:-auto}" == "1" ]]; then
    want_pause=1
  elif [[ "${KEEP_OPEN:-auto}" == "0" ]]; then
    want_pause=0
  elif [[ "${OSTYPE:-}" == msys || "${OSTYPE:-}" == cygwin ]]; then
    want_pause=1
  fi
  [[ "$want_pause" -eq 0 ]] && return 0
  echo
  if [[ -e /dev/tty ]]; then
    read -rp "Presiona Enter para cerrar..." _ </dev/tty
  elif [[ -t 0 ]]; then
    read -rp "Presiona Enter para cerrar..." _
  else
    echo "Presiona Ctrl+C para cerrar... (cerrando automáticamente en 120s)"
    sleep 120 || true
  fi
}
trap pause_if_needed EXIT

# ==================== Modo PC (Conda) ===================
load_conda_like() {
  if is_cmd conda; then
    eval "$("$(command -v conda)" shell.bash hook)"
    return 0
  fi
  if [[ -n "${CONDA_EXE:-}" ]]; then
    eval "$("$CONDA_EXE" shell.bash hook)"
    return 0
  fi
  for CAND in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"
  do
    [[ -f "$CAND" ]] && { source "$CAND"; return 0; }
  done
  if is_cmd micromamba; then
    eval "$(micromamba shell hook -s bash)"
    return 0
  fi
  return 1
}

create_env_if_needed_pc() {
  if is_cmd conda; then
    if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
      echo "Creando entorno $ENV_NAME (Conda) con Python $PYTHON_VER..."
      conda create -y -n "$ENV_NAME" python="$PYTHON_VER"
    fi
  else
    if ! micromamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
      echo "Creando entorno $ENV_NAME (Micromamba) con Python $PYTHON_VER..."
      micromamba create -y -n "$ENV_NAME" python="$PYTHON_VER"
    fi
  fi
}

run_in_env_pc() {
  if is_cmd conda; then
    conda run --no-capture-output -n "$ENV_NAME" "$@"
  else
    micromamba run -n "$ENV_NAME" "$@"
  fi
}

# =================== Modo Jetson (sin Conda) ===================
py_jetson() { python3 "$@"; }
pip_jetson() { python3 -m pip "$@"; }

install_jetson_system_prereqs() {
  [[ "$IS_JETSON" -eq 1 ]] || return 0
  echo "Detectado Jetson (aarch64). Instalando prerrequisitos del sistema..."
  sudo apt-get update
  sudo apt-get install -y python3-pip python3-opencv python3-pyqt5
  sudo apt-get install -y libopenblas-base libatlas-base-dev || true
  # ⚙️ Solución al error 'libomp.so'
  sudo apt-get install -y libomp5 libomp-dev
  python3 -m pip install --upgrade pip setuptools wheel
}

ensure_torch_jetson() {
  if py_jetson - <<'PY'
try:
    import torch
    import sys
    ok = torch.__version__.startswith(("1.10.","1.11"))
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
PY
  then
    echo "PyTorch (Jetson) ya instalado: $(python3 -c 'import torch; print(torch.__version__)')"
    return 0
  fi

  echo "Instalando PyTorch (Jetson JP4.x, wheel NVIDIA)…"
  local WHEEL_URL="https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/torch-1.11.0a0+17540c5+nv22.01-cp36-cp36m-linux_aarch64.whl"
  pip_jetson install --no-cache-dir --no-deps "$WHEEL_URL"

  echo "Verificando PyTorch…"
  py_jetson - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu  :", torch.cuda.get_device_name(0))
PY
}

# ================== Requirements ======================
install_requirements_pc() {
  [[ -f requirements.txt ]] || { echo "ADVERTENCIA: no hay requirements.txt (PC)."; return 0; }
  if [[ -n "$TORCH_EXTRA_INDEX_URL" ]]; then
    export PIP_EXTRA_INDEX_URL="$TORCH_EXTRA_INDEX_URL"
  fi
  echo "Instalando dependencias (PC, conda/micromamba)…"
  run_in_env_pc python -m pip install --upgrade pip
  run_in_env_pc python -m pip install -r requirements.txt
}

install_requirements_jetson() {
  [[ -f requirements.txt ]] || { echo "ADVERTENCIA: no hay requirements.txt (Jetson)."; return 0; }
  echo "Instalando dependencias (Jetson, pip3 del sistema)…"
  TMP_REQ="$(mktemp)"
  grep -Ev '^[[:space:]]*(torch|torchaudio|torchvision)($|[=<>!~])' requirements.txt > "$TMP_REQ" || true
  pip_jetson install --no-cache-dir -r "$TMP_REQ" || {
    echo "Reintentando dependencias (Jetson)…"
    PIP_DISABLE_PIP_VERSION_CHECK=1 pip_jetson install --retries 5 --timeout 180 --no-cache-dir -r "$TMP_REQ"
  }
  rm -f "$TMP_REQ"
}

# ===================== MAIN =============================
case "${1:-}" in
  env)
    if [[ "$IS_JETSON" -eq 1 ]]; then
      install_jetson_system_prereqs
      ensure_torch_jetson
      install_requirements_jetson
    else
      if ! load_conda_like; then
        echo "ERROR: No se encontró conda/micromamba en PC."
        exit 2
      fi
      create_env_if_needed_pc
      install_requirements_pc
    fi
    ;;
  deps)
    if [[ "$IS_JETSON" -eq 1 ]]; then
      install_jetson_system_prereqs
      ensure_torch_jetson
      install_requirements_jetson
    else
      if ! load_conda_like; then echo "ERROR: falta conda/micromamba en PC."; exit 2; fi
      create_env_if_needed_pc
      install_requirements_pc
    fi
    ;;
  check)
    if [[ "$IS_JETSON" -eq 1 ]]; then
      python3 - <<'PY'
import torch, cv2, open3d as o3d
print("torch:", torch.__version__, "cv2:", cv2.__version__, "open3d:", o3d.__version__)
PY
    else
      if ! load_conda_like; then echo "ERROR: falta conda/micromamba en PC."; exit 2; fi
      run_in_env_pc python - <<'PY'
import torch, cv2, open3d as o3d
print("torch:", torch.__version__, "cv2:", cv2.__version__, "open3d:", o3d.__version__)
PY
    fi
    ;;
  *)
    echo "Uso: $0 {env|deps|check}"
    ;;
esac
