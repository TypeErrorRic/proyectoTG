#!/usr/bin/env bash
set -euo pipefail

# ================== Config por defecto ==================
ENV_NAME="${ENV_NAME:-TG}"

# PC moderno -> 3.10; en Jetson (JP4.x) se ignorará y usaremos python3 del sistema
PYTHON_VER="${PYTHON_VER:-3.10}"

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
# En Jetson usamos python3/pip3 del sistema (JP4.x → Python 3.6)
py_jetson() { python3 "$@"; }
pip_jetson() { python3 -m pip "$@"; }

install_jetson_system_prereqs() {
  [[ "$IS_JETSON" -eq 1 ]] || return 0
  echo "Detectado Jetson (aarch64). Instalando prerrequisitos del sistema..."
  sudo apt-get update
  sudo apt-get install -y python3-pip python3-opencv python3-pyqt5
  sudo apt-get install -y libopenblas-base libatlas-base-dev || true
  # Asegurar pip moderno suficiente para wheel local
  python3 -m pip install --upgrade pip setuptools wheel
}

ensure_torch_jetson() {
  # Instala SOLO torch (wheel NVIDIA). torchvision/torchaudio son opcionales/complicados en JP4.
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
  # Wheel estable publicado para JP4.6.x (Python 3.6 ABI cp36m)
  local WHEEL_URL="https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/torch-1.11.0a0+17540c5+nv22.01-cp36-cp36m-linux_aarch64.whl"
  pip_jetson install --no-cache-dir --no-deps "$WHEEL_URL"

  echo "Verificando PyTorch…"
  py_jetson - <<'PY'
import torch, sys
print("torch:", torch.__version__)
print("cuda :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu  :", torch.cuda.get_device_name(0))
PY
}

# ================== Requirements ======================
install_requirements_pc() {
  [[ -f requirements.txt ]] || { echo "ADVERTENCIA: no hay requirements.txt (PC)."; return 0; }

  # Permite pasar índice CUDA para PC
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
  # Filtra torch/torchaudio/torchvision para no pisar el wheel de NVIDIA
  # Nota: mantenemos otras libs tal cual.
  TMP_REQ="$(mktemp)"
  grep -Ev '^[[:space:]]*(torch|torchaudio|torchvision)($|[=<>!~])' requirements.txt > "$TMP_REQ" || true
  pip_jetson install --no-cache-dir -r "$TMP_REQ" || {
    echo "Reintentando dependencias (Jetson)…"
    PIP_DISABLE_PIP_VERSION_CHECK=1 pip_jetson install --retries 5 --timeout 180 --no-cache-dir -r "$TMP_REQ"
  }
  rm -f "$TMP_REQ"
}

# ================== Info de entorno ===================
print_env_info_pc() {
  echo "================ ENV INFO (PC) ================"
  echo "Entorno : $ENV_NAME"
  echo "Plataf. : $(uname -m)"
  echo "Python  : $(run_in_env_pc python -V 2>&1)"
  echo "Ruta py : $(run_in_env_pc python -c 'import sys;print(sys.executable)')"
  echo "PIP     : $(run_in_env_pc python -c 'import pip;print(pip.__version__)')"
  run_in_env_pc python - <<'PY'
try:
    import torch
    print(f"PyTorch : {torch.__version__}")
    import torch as _t
    print("CUDA    :", _t.cuda.is_available())
except Exception:
    print("PyTorch : no instalado")
PY
  echo "==============================================="
}

print_env_info_jetson() {
  echo "================ ENV INFO (Jetson) ================"
  echo "Plataf. : $(uname -m)  (Jetson=1)"
  echo "Python  : $(python3 -V 2>&1)"
  echo "Ruta py : $(python3 - <<'PY'
import sys; print(sys.executable)
PY
)"
  echo "PIP     : $(python3 -m pip -V 2>&1)"
  python3 - <<'PY'
try:
    import torch
    print(f"PyTorch : {torch.__version__}")
    print("CUDA    :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU     :", torch.cuda.get_device_name(0))
        import torch as _t
        print("CUDA ver:", getattr(_t.version, "cuda", "desconocida"))
except Exception:
    print("PyTorch : no instalado")
PY
  echo "==================================================="
}

# ================== Autotest de imports ===============
check_imports_pc() {
  run_in_env_pc python - <<'PY'
mods = {
  "torch": "import torch; print('torch', torch.__version__)",
  "cv2": "import cv2; print('opencv', cv2.__version__)",
  "open3d": "import open3d as o3d; print('open3d', o3d.__version__)",
}
print("============= CHECK IMPORTS (PC) =============")
for name, code in mods.items():
  try:
    exec(code)
  except Exception as e:
    print(f"{name}  !! ERROR -> {e}")
print("=============================================")
PY
}

check_imports_jetson() {
  python3 - <<'PY'
mods = {
  "torch": "import torch; print('torch', torch.__version__)",
  "cv2": "import cv2; print('opencv', cv2.__version__)",
  "open3d": "import open3d as o3d; print('open3d', o3d.__version__)",
}
print("=========== CHECK IMPORTS (Jetson) ===========")
for name, code in mods.items():
  try:
    exec(code)
  except Exception as e:
    print(f"{name}  !! ERROR -> {e}")
print("=============================================")
PY
}

# ===================== MAIN =============================
case "${1:-}" in
  env)
    if [[ "$IS_JETSON" -eq 1 ]]; then
      install_jetson_system_prereqs
      ensure_torch_jetson
      install_requirements_jetson
      print_env_info_jetson
    else
      if ! load_conda_like; then
        echo "ERROR: No se encontró conda/micromamba en PC."
        exit 2
      fi
      create_env_if_needed_pc
      install_requirements_pc
      print_env_info_pc
    fi
    if [[ "${2:-}" == "full" ]]; then
      if [[ "$IS_JETSON" -eq 1 ]]; then
        echo "============= PIP LIST (Jetson) ============="
        python3 -m pip list
        echo "============================================="
      else
        echo "============= PIP LIST ($ENV_NAME) =========="
        run_in_env_pc python -m pip list
        echo "============= CONDA LIST ($ENV_NAME) ========"
        if is_cmd conda; then conda list -n "$ENV_NAME" || true; else micromamba list -n "$ENV_NAME" || true; fi
        echo "============================================="
      fi
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
      check_imports_jetson
    else
      if ! load_conda_like; then echo "ERROR: falta conda/micromamba en PC."; exit 2; fi
      check_imports_pc
    fi
    ;;
  shell)
    if [[ "$IS_JETSON" -eq 1 ]]; then
      echo "Abriendo subshell (Jetson, sin conda)… Usa 'exit' para salir."
      bash --noprofile --norc -i
    else
      if ! load_conda_like; then echo "ERROR: falta conda/micromamba en PC."; exit 2; fi
      echo "Abriendo subshell con entorno '$ENV_NAME' (PC)…"
      if is_cmd conda; then
        conda activate "$ENV_NAME" 2>/dev/null || true
      else
        micromamba activate "$ENV_NAME" 2>/dev/null || true
      fi
      bash --noprofile --norc -i <<'BASH'
echo "($ENV_NAME) listo. Escribe 'exit' para volver."
export PS1="($ENV_NAME) $PS1"
BASH
    fi
    ;;
  train)
    echo "Entrenando modelo..."
    if [[ "$IS_JETSON" -eq 1 ]]; then
      install_jetson_system_prereqs
      ensure_torch_jetson
      install_requirements_jetson
      python3 src/train.py --config configs/unet_baseline.yaml
    else
      if ! load_conda_like; then echo "ERROR: falta conda/micromamba en PC."; exit 2; fi
      create_env_if_needed_pc
      install_requirements_pc
      run_in_env_pc python src/train.py --config configs/unet_baseline.yaml
    fi
    ;;
  infer)
    echo "Ejecutando inferencia..."
    INP="${2:-}"; OUT="${3:-}"
    if [[ -z "$INP" || -z "$OUT" ]]; then
      echo "Uso: $0 infer <input_depth.png> <output_mask.png>"
      exit 1
    fi
    if [[ "$IS_JETSON" -eq 1 ]]; then
      install_jetson_system_prereqs
      ensure_torch_jetson
      install_requirements_jetson
      python3 src/infer.py --input "$INP" --output "$OUT"
    else
      if ! load_conda_like; then echo "ERROR: falta conda/micromamba en PC."; exit 2; fi
      create_env_if_needed_pc
      install_requirements_pc
      run_in_env_pc python src/infer.py --input "$INP" --output "$OUT"
    fi
    ;;
  visual)
    echo "Ejecutando visualizador..."
    cd src/data
    if [[ "$IS_JETSON" -eq 1 ]]; then
      export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
      python3 visualizador.py
    else
      if ! load_conda_like; then echo "ERROR: falta conda/micromamba en PC."; exit 2; fi
      run_in_env_pc python visualizador.py
    fi
    ;;
  *)
    cat <<EOF
Uso: $0 {env|env full|deps|check|shell|train|infer|visual} [args]
  env               Prepara entorno (PC: conda | Jetson: sistema), instala requirements y muestra info.
  env full          Además lista paquetes.
  deps              (Re)instala dependencias.
  check             Autotest de imports.
  shell             Subshell (PC: conda activado | Jetson: shell normal).
  train             Entrena (usa configs/unet_baseline.yaml).
  infer <in> <out>  Ejecuta inferencia.
  visual            Lanza visualizador de ejemplo (src/data/visualizador.py).
Variables útiles:
  ENV_NAME=...           (PC: nombre del entorno conda; defecto: TG)
  PYTHON_VER=...         (PC: 3.10; Jetson ignora y usa python3 del sistema)
  KEEP_OPEN=1            Pausa al terminar (útil en Git Bash/Windows)
  TORCH_EXTRA_INDEX_URL  Índice extra de PyTorch (p.ej. CUDA en PC)
EOF
    exit 1
    ;;
esac
