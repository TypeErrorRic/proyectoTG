#!/usr/bin/env bash
set -euo pipefail

# ================== Config por defecto ==================
ENV_NAME="${ENV_NAME:-TG}"

# PC moderno -> 3.10; Jetson JP4.x -> 3.8 (se fuerza más abajo si detectamos Jetson)
PYTHON_VER="${PYTHON_VER:-3.10}"

# Puedes apuntar a índices de PyTorch si quieres CUDA en PC:
# ej: TORCH_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"
TORCH_EXTRA_INDEX_URL="${TORCH_EXTRA_INDEX_URL:-}"

# ========================================================
# Detectores de plataforma
is_cmd() { command -v "$1" >/dev/null 2>&1; }

is_aarch64() { [[ "$(uname -m)" == "aarch64" ]]; }
is_jetson_board() {
  # Señales típicas de Jetson/JetPack
  [[ -f /etc/nv_tegra_release ]] && return 0
  if is_cmd nvidia-smi; then
    # (algunos Jetson no traen nvidia-smi)
    return 0
  fi
  # Tegra en el árbol de dispositivos
  [[ -d /proc/device-tree/tegra-fuse || -d /proc/device-tree/chosen/nvidia,tegra-udrm ]] && return 0
  return 1
}

IS_JETSON=0
if is_aarch64 && is_jetson_board; then
  IS_JETSON=1
fi

# Jetson típico (JetPack 4.x) usa mejor Python 3.8
if [[ "$IS_JETSON" -eq 1 && "${PYTHON_VER}" == "3.10" ]]; then
  PYTHON_VER="3.8"
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

# ============== Cargar conda / micromamba ===============
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
    if [[ -f "$CAND" ]]; then
      # shellcheck disable=SC1090
      source "$CAND"
      return 0
    fi
  done
  # Intento con micromamba si existe
  if is_cmd micromamba; then
    eval "$(micromamba shell hook -s bash)"
    return 0
  fi
  return 1
}

if ! load_conda_like; then
  echo "ERROR: No se encontró conda/micromamba. Instala Miniconda o Micromamba e inténtalo de nuevo."
  exit 2
fi

# ============ Crear entorno si no existe =================
create_env_if_needed() {
  # Ya no usamos --system-site-packages con conda/micromamba (no es válido)
  if is_cmd conda; then
    if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
      echo "Creando entorno $ENV_NAME con Python $PYTHON_VER..."
      conda create -y -n "$ENV_NAME" python="$PYTHON_VER"
    fi
  else
    if ! micromamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
      echo "Creando entorno (micromamba) $ENV_NAME con Python $PYTHON_VER..."
      micromamba create -y -n "$ENV_NAME" python="$PYTHON_VER"
    fi
  fi
}

run_in_env() {
  if is_cmd conda; then
    conda run --no-capture-output -n "$ENV_NAME" "$@"
  else
    micromamba run -n "$ENV_NAME" "$@"
  fi
}

# ============== Prerrequisitos Jetson (APT) =============
install_jetson_system_prereqs() {
  [[ "$IS_JETSON" -eq 1 ]] || return 0
  echo "Detectado Jetson (aarch64). Instalando prerrequisitos del sistema..."
  sudo apt-get update
  # OpenCV y PyQt5 del sistema (evita compilar y problemas de wheels)
  sudo apt-get install -y python3-opencv python3-pyqt5
  # BLAS/Atlas, útiles para NumPy/Scipy
  sudo apt-get install -y libopenblas-base libatlas-base-dev || true
}

# ============== Torch según plataforma (Jetson Nano 4GB JP4.x) ==================
ensure_torch() {
  if [[ "$IS_JETSON" -eq 1 ]]; then
    # Verifica que estén las versiones compatibles de NVIDIA para JP 4.x
    if ! run_in_env python - <<'PY'
import sys
try:
    import torch, torchvision, torchaudio
    ok = torch.__version__.startswith("1.10.") and \
         torchvision.__version__.startswith("0.11.") and \
         torchaudio.__version__.startswith("0.10.")
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
PY
    then
      echo "Instalando PyTorch (Jetson Nano JP4.x, wheels oficiales NVIDIA)…"

      # Validar que el entorno usa Python 3.8 (JP4.x)
      JPY=$(run_in_env python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)
      if [[ "$JPY" != "3.8" ]]; then
        echo "[Torch] En Jetson JP4.x necesitas Python 3.8 (actual: $JPY)."
        echo "        Recrea el entorno: conda remove -n $ENV_NAME --all -y && conda create -n $ENV_NAME python=3.8 -y"
        return 1
      fi

      # URLs directas a las ruedas NVIDIA para JetPack 4.x (repo jp/v46), Python 3.8 (cp38-cp38)
      BASE="https://developer.download.nvidia.com/compute/redist/jp/v46/pytorch"
      TORCH_WHL="torch-1.10.0%2Bnv22.02-cp38-cp38-linux_aarch64.whl"
      TV_WHL="torchvision-0.11.1%2Bnv22.02-cp38-cp38-linux_aarch64.whl"
      TA_WHL="torchaudio-0.10.0%2Bnv22.02-cp38-cp38-linux_aarch64.whl"

      # Instalar cada wheel por URL directa (evita PyPI)
      run_in_env python -m pip install --no-cache-dir --no-deps "${BASE}/${TORCH_WHL}"
      run_in_env python -m pip install --no-cache-dir --no-deps "${BASE}/${TV_WHL}"
      run_in_env python -m pip install --no-cache-dir --no-deps "${BASE}/${TA_WHL}"

      # Verificación
      run_in_env python - <<'PY'
import torch, torchvision, torchaudio
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
PY
    else
      echo "PyTorch (Jetson) ya está en versión compatible."
    fi
  else
    # En PC deja que requirements instale torch>=2.x (opcional índice extra)
    if [[ -n "$TORCH_EXTRA_INDEX_URL" ]]; then
      export PIP_EXTRA_INDEX_URL="$TORCH_EXTRA_INDEX_URL"
    fi
  fi
}



# ============== Instalar requirements ===================
ensure_requirements() {
  if [[ ! -f requirements.txt ]]; then
    echo "ADVERTENCIA: no se encontró requirements.txt en $(pwd). Omitiendo instalación."
    return 0
  fi

  echo "Instalando/verificando dependencias de requirements.txt en '$ENV_NAME'..."
  # 1) Torch primero (según plataforma) para evitar que el resolver se trabe
  ensure_torch

  # 2) Instalar el resto. Evitamos que intente reinstalar Torch/torchaudio/torchvision.
  #    NOTA: esto asume que tu requirements usa marcadores PEP-508 como te propuse.
  local pip_args=( -v --no-cache-dir -r requirements.txt )
  # No-deps parcial: usamos --no-deps y luego resolvemos dependencias faltantes si fuese necesario.
  # En la práctica, con marcadores correctos, no suele hacer falta. Dejamos normal:
  if ! run_in_env python -m pip install "${pip_args[@]}"; then
    echo "Primer intento falló. Reintentando con más tiempo y reintentos..."
    PIP_DISABLE_PIP_VERSION_CHECK=1 run_in_env python -m pip install --retries 5 --timeout 180 "${pip_args[@]}"
  fi

  # 3) pip check para verificar consistencia (ignoramos si falla por extras no críticos)
  if run_in_env python -m pip --help >/dev/null 2>&1; then
    if run_in_env python -m pip check >/dev/null 2>&1; then
      echo "Dependencias consistentes (pip check OK)."
    else
      echo "NOTA: pip check reportó conflictos; puedes compartir el log si necesitas que los fijemos."
    fi
  fi
}

# ============== Info de entorno =========================
print_env_info() {
  echo "================ ENV INFO ================"
  echo "Entorno : $ENV_NAME"
  echo "Plataf. : $(uname -m)  (Jetson=$IS_JETSON)"
  echo "Python  : $(run_in_env python -V 2>&1)"
  echo "Ruta py : $(run_in_env python -c 'import sys;print(sys.executable)')"
  echo "PIP     : $(run_in_env python -c 'import pip;print(pip.__version__)')"
  run_in_env python - <<'PY'
try:
    import torch
    print(f"PyTorch : {torch.__version__}")
    if torch.cuda.is_available():
        print("CUDA    : disponible")
        try:
            print(f"GPU     : {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
        import torch as _t
        print(f"CUDA ver: {getattr(_t.version, 'cuda', 'desconocida')}")
    else:
        print("CUDA    : NO disponible")
except Exception:
    print("PyTorch : no instalado")
PY
  echo "=========================================="
}

# ============== Listar paquetes =========================
print_pkg_lists() {
  echo "============= PIP LIST ($ENV_NAME) ============="
  run_in_env python -m pip list
  echo "============= CONDA LIST ($ENV_NAME) ==========="
  if is_cmd conda; then
    conda list -n "$ENV_NAME" || true
  else
    micromamba list -n "$ENV_NAME" || true
  fi
  echo "==============================================="
}

# ============== Autotest de imports =====================
check_imports() {
  run_in_env python - <<'PY'
mods = {
  "torch": "import torch; print('torch', torch.__version__)",
  "torchvision": "import torchvision; print('torchvision', torchvision.__version__)",
  "torchaudio": "import torchaudio; print('torchaudio', torchaudio.__version__)",
  "cv2": "import cv2; print('opencv', cv2.__version__)",
  "open3d": "import open3d as o3d; print('open3d', o3d.__version__)",
  "h5py": "import h5py; print('h5py', h5py.__version__)",
  "sklearn": "import sklearn; print('sklearn', sklearn.__version__)",
  "albumentations": "import albumentations as A; import importlib.metadata as im; print('albumentations', im.version('albumentations'))",
  "torchmetrics": "import torchmetrics as tm; print('torchmetrics', tm.__version__)",
  "segmentation_models_pytorch": "import importlib.metadata as im; print('segmentation-models-pytorch', im.version('segmentation-models-pytorch'))",
}
print("============= CHECK IMPORTS =============")
for name, code in mods.items():
  try:
    exec(code)
  except Exception as e:
    print(f"{name}  !! ERROR -> {e}")
print("=========================================")
PY
}

# ===================== MAIN =============================
create_env_if_needed
install_jetson_system_prereqs

case "${1:-}" in
  env)
    ensure_requirements
    print_env_info
    if [[ "${2:-}" == "full" ]]; then
      print_pkg_lists
    fi
    ;;
  deps)
    ensure_requirements
    ;;
  visual)
    # Ejecuta un visualizador de ejemplo
    if is_cmd conda; then eval "$("$(command -v conda)" shell.bash hook)"; fi
    conda activate "$ENV_NAME" 2>/dev/null || true
    export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
    cd src/data
    python visualizador.py
    ;;
  check)
    check_imports
    ;;
  shell)
    echo "Abriendo subshell con entorno '$ENV_NAME' activado..."
    if is_cmd conda; then eval "$("$(command -v conda)" shell.bash hook)"; fi
    conda activate "$ENV_NAME" 2>/dev/null || true
    bash --noprofile --norc -i <<'BASH'
if command -v conda >/dev/null 2>&1; then
  eval "$("$(command -v conda)" shell.bash hook)"
  conda activate "$ENV_NAME" 2>/dev/null || true
fi
echo "($ENV_NAME) listo. Escribe 'exit' para volver."
export PS1="($ENV_NAME) $PS1"
BASH
    ;;
  train)
    echo "Entrenando modelo..."
    ensure_requirements
    run_in_env python src/train.py --config configs/unet_baseline.yaml
    ;;
  infer)
    echo "Ejecutando inferencia..."
    ensure_requirements
    INP="${2:-}"; OUT="${3:-}"
    if [[ -z "$INP" || -z "$OUT" ]]; then
      echo "Uso: $0 infer <input_depth.png> <output_mask.png>"
      exit 1
    fi
    run_in_env python src/infer.py --input "$INP" --output "$OUT"
    ;;
  *)
    cat <<EOF
Uso: $0 {env|env full|deps|check|shell|train|infer} [args]
  env               Crea/verifica entorno, instala requirements y muestra info.
  env full          Además lista paquetes (pip/conda).
  deps              (Re)instala/verifica dependencias de requirements.txt.
  check             Autotest de imports y versiones clave.
  shell             Subshell con el entorno activado.
  train             Entrena (usa configs/unet_baseline.yaml).
  infer <in> <out>  Ejecuta inferencia.
Variables útiles:
  ENV_NAME=...           (por defecto: TG)
  PYTHON_VER=...         (PC: 3.10 | Jetson: 3.8 por defecto)
  KEEP_OPEN=1            Pausa al terminar (útil en Git Bash/Windows).
  TORCH_EXTRA_INDEX_URL  Índice extra de PyTorch (p.ej. CUDA en PC).
EOF
    exit 1
    ;;
esac
