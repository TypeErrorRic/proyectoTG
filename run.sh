#!/usr/bin/env bash
set -euo pipefail

# ================== Config por defecto ==================
ENV_NAME="${ENV_NAME:-TG_develop}"  # Nombre del entorno conda en PC
PYTHON_VER="3.6"                    # Python 3.6 en PC y Jetson

# ================== Detectores de plataforma ==================
is_cmd() { command -v "$1" >/dev/null 2>&1; }
is_aarch64() { [[ "$(uname -m)" == "aarch64" ]]; }
is_linux() { [[ "${OSTYPE:-}" == linux* ]]; }
is_msys() { [[ "${OSTYPE:-}" == msys || "${OSTYPE:-}" == cygwin ]]; }  # Git Bash
is_darwin() { [[ "${OSTYPE:-}" == darwin* ]]; }
is_windows_shell() { is_msys; }
is_jetson_board() {
  [[ -f /etc/nv_tegra_release ]] && return 0
  [[ -d /proc/device-tree/tegra-fuse || -d /proc/device-tree/chosen/nvidia,tegra-udrm ]] && return 0
  return 1
}

IS_JETSON=0
if is_aarch64 && is_jetson_board; then IS_JETSON=1; fi

# ================== Pausa inteligente ===================
pause_if_needed() {
  local want_pause=0
  if [[ "${KEEP_OPEN:-auto}" == "1" ]]; then
    want_pause=1
  elif [[ "${KEEP_OPEN:-auto}" == "0" ]]; then
    want_pause=0
  elif is_windows_shell; then
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

# ==================== PC (Conda) ====================
load_conda() {
  if is_cmd conda; then
    eval "$("$(command -v conda)" shell.bash hook)"
    return 0
  fi
  for CAND in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"
  do
    [[ -f "$CAND" ]] && { # shellcheck disable=SC1090
      source "$CAND"; return 0;
    }
  done
  return 1
}

create_env_pc() {
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Creando entorno $ENV_NAME (Conda) con Python $PYTHON_VER…"
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VER"
  fi
}

run_in_env_pc() {
  conda run --no-capture-output -n "$ENV_NAME" "$@"
}

install_requirements_pc() {
  [[ -f requirements.txt ]] || { echo "ADVERTENCIA: no hay requirements.txt (PC)."; return 0; }
  echo "Instalando dependencias (PC, conda env: $ENV_NAME)…"
  run_in_env_pc python -m pip install --upgrade "pip<22" "setuptools<60" "wheel<0.38"
  run_in_env_pc python -m pip install opencv-python==4.5.5.64
  run_in_env_pc python -m pip install -r requirements.txt
}

install_realsense_pc_in_env() {
  # Instala pyrealsense2 dentro del entorno (Windows/Linux x86_64). macOS no soportado.
  if is_darwin; then
    echo "Aviso: macOS no está soportado oficialmente por Intel RealSense. Saltando instalación."
    return 0
  fi

  echo "Instalando pyrealsense2 (PC) en el entorno conda $ENV_NAME…"

  # ¿Ya está instalado? (solo validamos que importe)
  if run_in_env_pc python - <<'PY'
try:
    import pyrealsense2 as rs
    print("pyrealsense2 ya importable en este entorno.")
    import sys; sys.exit(0)
except Exception as e:
    print("pyrealsense2 NO importable todavía:", repr(e))
    import sys; sys.exit(1)
PY
  then
    :
  else
    # Instalar wheel desde PyPI (para Py3.6 toma una versión compatible)
    run_in_env_pc python -m pip install pyrealsense2
  fi

  echo "Verificación de import y detalle de instalación…"
  # No usamos rs.__version__ porque a veces no existe; tomamos versión de pkg_resources
  run_in_env_pc python - <<'PY'
import sys
try:
    import pyrealsense2 as rs
    print("Import OK  ->", rs.__file__)
    try:
        import pkg_resources
        v = pkg_resources.get_distribution("pyrealsense2").version
        print("Version    ->", v)
    except Exception:
        # Fallback: pip show
        import subprocess, shlex
        try:
            out = subprocess.check_output(shlex.split(sys.executable + " -m pip show pyrealsense2")).decode(errors="ignore")
            for line in out.splitlines():
                if line.startswith("Version:"):
                    print(line)
                    break
        except Exception:
            print("Version    -> (no disponible)")
except Exception as e:
    print("FALLO import pyrealsense2:", repr(e))
    sys.exit(2)
PY
}

# =================== Jetson (sin conda) ===================
pip_jetson() { python3 -m pip "$@"; }

install_jetson_system_prereqs() {
  echo "Instalando prerrequisitos del sistema (Jetson)…"
  sudo apt-get update
  sudo apt-get install -y python3-pip python3-opencv
  # BLAS y OpenMP
  sudo apt-get install -y libopenblas-base libatlas-base-dev || true
  sudo apt-get install -y libomp5 libomp-dev
  # Herramientas de build y dependencias librealsense
  sudo apt-get install -y libssl-dev libusb-1.0-0-dev pkg-config \
                          libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
                          cmake git udev
  # Pip/setuptools compatibles con Python 3.6
  python3 -m pip install --upgrade "pip<22" "setuptools<60" "wheel<0.38"
  # PATH local
  if ! grep -q 'export PATH="\$HOME/.local/bin' "$HOME/.bashrc" 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
  fi
  #Inastalar Matploid:
  sudo apt install -y python3-matplotlib
}

install_requirements_jetson() {
  [[ -f requirements.txt ]] || { echo "ADVERTENCIA: no hay requirements.txt (Jetson)."; return 0; }
  echo "Instalando dependencias (Jetson, pip3 del sistema)…"
  TMP_REQ="$(mktemp)"
  # Evitar torch* por incompatibilidades
  grep -Ev '^[[:space:]]*(torch|torchaudio|torchvision)($|[=<>!~])' requirements.txt > "$TMP_REQ" || true
  PIP_DISABLE_PIP_VERSION_CHECK=1 pip_jetson install --no-cache-dir -r "$TMP_REQ"
  rm -f "$TMP_REQ"
}

install_realsense_jetson_in_env() {
  echo "==> Instalando pyrealsense2 (Jetson) con pybind11 2.10.4 (compatible Py3.6)…"

  # Asegura prerequisitos del sistema (si ya lo haces fuera, puedes omitir esta línea)
  install_jetson_system_prereqs

  # Reglas udev + repo librealsense
  [[ -d librealsense ]] || git clone https://github.com/IntelRealSense/librealsense.git
  sudo cp librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/ || true
  sudo udevadm control --reload-rules && sudo udevadm trigger

  # ===== Pin pybind11 compatible con Python 3.6 =====
  python3 -m pip install --upgrade "pip<22" "setuptools<60" "wheel<0.38"
  python3 -m pip install "pybind11==2.10.4"
  PYBIND11_DIR="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')"
  echo "pybind11_DIR: $PYBIND11_DIR"

  # ===== Configurar y compilar librealsense apuntando al pybind11 externo =====
  pushd librealsense >/dev/null
  rm -rf build
  mkdir -p build && cd build

  # Nota: FORZAMOS RSUSB para no parchear kernel; bindings Python ON; Py3.6 explícito.
  if ! cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DFORCE_RSUSB_BACKEND=ON \
      -DBUILD_PYTHON_BINDINGS=ON \
      -DPYTHON_EXECUTABLE="$(which python3)" \
      -Dpybind11_DIR="${PYBIND11_DIR}" \
      -DPYBIND11_PYTHON_VERSION=3.6 \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_GRAPHICAL_EXAMPLES=OFF; then
    echo "ERROR: CMake falló aun con pybind11_DIR=${PYBIND11_DIR}"
    echo "Revisa librealsense/build/CMakeFiles/CMakeOutput.log para más detalles."
    popd >/dev/null
    return 2
  fi

  echo "Compilando librealsense… (puede tardar)"
  make -j2
  sudo make install
  sudo ldconfig
  popd >/dev/null

  echo "Verificando import de pyrealsense2…"
  python3 - <<'PY'
try:
    import pyrealsense2 as rs
    print("pyrealsense2 OK:", getattr(rs, "__file__", "(sin ruta)"))
except Exception as e:
    print("Fallo import pyrealsense2:", repr(e))
    raise
PY
}

# ===================== MAIN =============================
case "${1:-}" in
  env|deps)
    if [[ "$IS_JETSON" -eq 1 ]]; then
      # Jetson: instala deps + requirements + compila librealsense (pyrealsense2)
      install_jetson_system_prereqs
      install_requirements_jetson
      install_realsense_jetson_in_env
    else
      # PC: crea/actualiza entorno + requirements + pyrealsense2 en el entorno
      if ! load_conda; then echo "ERROR: No se encontró conda en PC."; exit 2; fi
      create_env_pc
      install_requirements_pc
      install_realsense_pc_in_env
    fi
    ;;

  check)
    if [[ "$IS_JETSON" -eq 1 ]]; then
      python3 - <<'PY'
try:
    import sys, platform, cv2
    print("Python:", sys.version.split()[0], "| Platform:", platform.machine())
    print("cv2   :", cv2.__version__)
except Exception as e:
    print("ERROR en check (Jetson):", e)
    raise
PY
    else
      if ! load_conda; then echo "ERROR: No se encontró conda en PC."; exit 2; fi
      run_in_env_pc python - <<'PY'
try:
    import sys, platform, cv2
    print("Python:", sys.version.split()[0], "| Platform:", platform.machine())
    print("cv2   :", cv2.__version__)
except Exception as e:
    print("ERROR en check (PC):", e)
    raise
PY
    fi
    ;;

    realsense-test)
    echo "Iniciando prueba de cámara (RealSense)…"
    if [[ "$IS_JETSON" -eq 1 ]]; then
      # === Jetson Nano ===
      if [[ -f "src/utilities/viewCamera.py" ]]; then
        echo "[Jetson] Ejecutando viewCamera.py con Python del sistema..."
        python3 src/utilities/viewCamera.py
      else
        echo "ERROR: No se encontró src/utilities/viewCamera.py"
        exit 1
      fi
    else
      # === PC (entorno conda) ===
      if ! load_conda; then echo "ERROR: No se encontró conda."; exit 2; fi
      if [[ -f "src/utilities/viewCamera.py" ]]; then
        echo "[PC] Ejecutando viewCamera.py dentro del entorno $ENV_NAME..."
        conda run --no-capture-output -n "$ENV_NAME" python src/utilities/viewCamera.py
      else
        echo "ERROR: No se encontró src/utilities/viewCamera.py"
        exit 1
      fi
    fi
    
    ;;

  visual)
    # Placeholder para tu visualizador propio
    echo "Usa 'realsense-test' para una prueba rápida de cámara RealSense."
    ;;

  *)
    echo "Uso: $0 {env|deps|check|realsense-test|visual}"
    ;;

esac
