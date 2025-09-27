#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-TG}"
PYTHON_VER="${PYTHON_VER:-3.10}"

# === Pausa inteligente =======================================================
pause_if_needed() {
  # Forzar pausa si KEEP_OPEN=1.
  # Nunca pausar si KEEP_OPEN=0.
  # Por defecto en Git Bash/Cygwin: pausar al final.
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
    # Leer directamente del TTY aunque stdin no sea interactivo
    read -rp "Presiona Enter para cerrar..." _ </dev/tty
  elif [[ -t 0 ]]; then
    read -rp "Presiona Enter para cerrar..." _
  else
    echo "Presiona Ctrl+C para cerrar... (cerrando automáticamente en 120s)"
    sleep 120 || true
  fi
}
trap pause_if_needed EXIT

# === 1) Cargar conda =========================================================
if command -v conda >/dev/null 2>&1; then
  eval "$("$(command -v conda)" shell.bash hook)"
elif [[ -n "${CONDA_EXE:-}" ]]; then
  eval "$("$CONDA_EXE" shell.bash hook)"
else
  for CAND in \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"
  do
    if [[ -f "$CAND" ]]; then
      # shellcheck disable=SC1090
      source "$CAND"
      break
    fi
  done
fi

# === 2) Crear entorno si no existe ==========================================
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Creando entorno $ENV_NAME con Python $PYTHON_VER..."
  conda create -y -n "$ENV_NAME" python="$PYTHON_VER"
fi

# === 3) Helper para ejecutar en el entorno sin activar =======================
run_in_env() {
  conda run --no-capture-output -n "$ENV_NAME" "$@"
}

# === 4) Instalar/verificar requirements.txt ==================================
ensure_requirements() {
  if [[ ! -f requirements.txt ]]; then
    echo "ADVERTENCIA: no se encontró requirements.txt en $(pwd). Omitiendo instalación."
    return 0
  fi

  echo "Instalando/verificando dependencias de requirements.txt en '$ENV_NAME'..."
  if ! run_in_env python -m pip install -r requirements.txt; then
    echo "Primer intento falló. Reintentando con --upgrade..."
    run_in_env python -m pip install --upgrade -r requirements.txt --use-deprecated=legacy-resolver
  fi

  if run_in_env python -m pip --help >/dev/null 2>&1; then
    if run_in_env python -m pip check >/dev/null 2>&1; then
      echo "Dependencias consistentes (pip check OK)."
    else
      echo "NOTA: pip check reportó conflictos; reintentando instalación..."
      run_in_env python -m pip install -r requirements.txt || true
    fi
  fi
}

# === 5) Info de entorno ======================================================
print_env_info() {
  echo "================ ENV INFO ================"
  echo "Entorno: $ENV_NAME"
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
        print(f"CUDA ver: {getattr(torch.version, 'cuda', 'desconocida')}")
    else:
        print("CUDA    : NO disponible")
except Exception:
    print("PyTorch : no instalado")
PY
  echo "=========================================="
}

# === 6) Listar paquetes ======================================================
print_pkg_lists() {
  echo "============= PIP LIST ($ENV_NAME) ============="
  run_in_env python -m pip list
  echo "============= CONDA LIST ($ENV_NAME) ==========="
  if command -v conda >/dev/null 2>&1; then
    conda list -n "$ENV_NAME" || true
  else
    echo "(conda no disponible para 'conda list')"
  fi
  echo "==============================================="
}

# === 7) Autotest de imports y versiones =====================================
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
  "segmentation_models_pytorch": "import segmentation_models_pytorch as smp; import importlib.metadata as im; print('segmentation-models-pytorch', im.version('segmentation-models-pytorch'))",
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

# === 8) CLI ==================================================================
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
  check)
    check_imports
    ;;
  shell)
    echo "Abriendo subshell con entorno '$ENV_NAME' activado..."
    if command -v conda >/dev/null 2>&1; then
      eval "$("$(command -v conda)" shell.bash hook)"
    fi
    conda activate "$ENV_NAME"
    bash --noprofile --norc -i <<'BASH'
if command -v conda >/dev/null 2>&1; then
  eval "$("$(command -v conda)" shell.bash hook)"
fi
conda activate "$ENV_NAME" 2>/dev/null || true
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
  env             Verifica/crea el entorno, instala requirements y muestra info.
  env full        Además lista paquetes (pip y conda).
  deps            (Re)instala/verifica dependencias de requirements.txt.
  check           Autotest de imports y versiones clave.
  shell           Abre una sub-shell con el entorno activado.
  train           Entrena el modelo (usa configs/unet_baseline.yaml).
  infer <in> <out>  Ejecuta inferencia.
Variables útiles:
  ENV_NAME=...   (por defecto: TG)
  PYTHON_VER=... (por defecto: 3.10)
  KEEP_OPEN=1    Fuerza pausa al terminar (útil en Git Bash/Windows).
EOF
    exit 1
    ;;
esac
