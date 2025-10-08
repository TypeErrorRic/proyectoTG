#!/usr/bin/env bash
set -euo pipefail

# ================== Configuración por defecto ==================
ENV_NAME="${ENV_NAME:-TG_py38}"  # Nombre del entorno conda (PC y Jetson)
PYTHON_VER="3.8"                 # Python 3.8 en ambos

# Tag de librealsense recomendado para JP4.x (estable)
RS_TAG="${RS_TAG:-v2.50.0}"

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

# ==================== Conda (carga/instala) ====================
load_conda() {
  if is_cmd conda; then
    eval "$("$(command -v conda)" shell.bash hook)"
    return 0
  fi
  for CAND in \
    "$HOME/miniforge/etc/profile.d/conda.sh" \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
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

install_miniforge_if_needed() {
  if load_conda; then return 0; fi
  echo "Conda no encontrado. Instalando Miniforge para este sistema…"
  local url=""
  if is_linux; then
    if is_aarch64; then
      url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"
    else
      url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    fi
  elif is_darwin; then
    url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-$(uname -m).sh"
  else
    echo "Plataforma no soportada automáticamente para instalar Miniforge."
    return 1
  fi
  mkdir -p "$HOME/.cache/miniforge"
  local shfile="$HOME/.cache/miniforge/Miniforge_installer.sh"
  curl -L "$url" -o "$shfile"
  bash "$shfile" -b -p "$HOME/miniforge"
  echo 'export PATH="$HOME/miniforge/bin:$PATH"' >> "$HOME/.bashrc"
  # shellcheck disable=SC1090
  source "$HOME/miniforge/etc/profile.d/conda.sh"
}

create_env_common() {
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Creando entorno $ENV_NAME (Conda) con Python $PYTHON_VER…"
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VER"
  fi
}

run_in_env() {
  conda run --no-capture-output -n "$ENV_NAME" "$@"
}

install_requirements_common() {
  [[ -f requirements.txt ]] || { echo "ADVERTENCIA: no hay requirements.txt."; return 0; }
  echo "Instalando dependencias (pip) dentro del entorno $ENV_NAME…"
  run_in_env python -m pip install --upgrade pip setuptools wheel
  run_in_env python -m pip install -r requirements.txt
  run_in_env python -m pip install opencv-python==4.8.1.78 
}

# =================== PC (x86_64) ===================
install_realsense_pc_in_env() {
  echo "Instalando pyrealsense2 (PC, wheel PyPI) dentro del entorno $ENV_NAME…"
  # Intento wheel
  if run_in_env python - <<'PY'
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
    run_in_env python -m pip install pyrealsense2 || true
  fi

  # Verificación; si falla el wheel, avisar.
  if ! run_in_env python - <<'PY'
try:
    import pyrealsense2 as rs, pkgutil
    print("Import OK  ->", rs.__file__)
except Exception as e:
    print("FALLO import pyrealsense2:", e)
    raise SystemExit(1)
PY
  then
    echo "ADVERTENCIA: No se pudo importar pyrealsense2 desde wheel en PC."
    echo "Puedes compilar desde fuente como en Jetson (mismo procedimiento) si lo necesitas en PC."
  fi
}

# =================== Jetson (aarch64) ===================
install_jetson_system_prereqs() {
  echo "Instalando prerrequisitos del sistema (Jetson)…"
  sudo apt-get update
  sudo apt-get install -y python3-pip
  sudo apt-get install -y libssl-dev libusb-1.0-0-dev pkg-config \
                          libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
                          cmake build-essential git udev
  # Opción útil en Nano para compilaciones largas (swap manual recomendado fuera de este script).
}

clone_rules_librealsense() {
  [[ -d librealsense ]] || git clone https://github.com/IntelRealSense/librealsense.git
  sudo cp librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/ || true
  sudo udevadm control --reload-rules && sudo udevadm trigger
}

compute_py_paths_in_env() {
  # Exporta variables PY, PY_INC, PY_LIB y CMAKE_PREFIX_PATH para el entorno activo
  export PY="$(conda run -n "$ENV_NAME" which python)"
  export PY_INC="$(conda run -n "$ENV_NAME" python - <<'PY'
import sysconfig; print(sysconfig.get_paths()["include"])
PY
  )"
  export PY_LIB="$(conda run -n "$ENV_NAME" python - <<'PY'
import sys,glob,sysconfig,os
ver=f"{sys.version_info.major}.{sys.version_info.minor}"
cands=[
  *glob.glob(f"{sys.prefix}/lib/libpython{ver}*.so"),
  *glob.glob(f"/usr/lib/aarch64-linux-gnu/libpython{ver}*.so"),
  *glob.glob(f"/usr/local/lib/libpython{ver}*.so"),
]
print(cands[0] if cands else "")
PY
  )"
  # Ayuda a CMake a encontrar pybind11 del entorno
  local PREFIX
  PREFIX="$(conda run -n "$ENV_NAME" python - <<'PY'
import sys; print(sys.prefix)
PY
  )"
  export CMAKE_PREFIX_PATH="${PREFIX}:${CMAKE_PREFIX_PATH:-}"
}

# Compila SOLO la lib C++ (sin bindings) y luego instala el wrapper con el Python del entorno conda
build_librealsense_cpp_then_wrapper_in_env() {
  echo "Compilando librealsense ${RS_TAG} (C++ solo) e instalando wrapper Python en '$ENV_NAME'…"

  # Reglas udev + repo
  [[ -d librealsense ]] || git clone https://github.com/IntelRealSense/librealsense.git
  sudo cp librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/ || true
  sudo udevadm control --reload-rules && sudo udevadm trigger

  pushd librealsense >/dev/null
  git fetch --tags || true
  git checkout "${RS_TAG}"

  # === 1) Compila SOLO la lib C++ (sin pybind) ===
  rm -rf build && mkdir build && cd build
  cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DFORCE_RSUSB_BACKEND=ON \
    -DBUILD_PYTHON_BINDINGS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_GRAPHICAL_EXAMPLES=OFF | tee cmake_config.log

  make -j2 || make -j1
  sudo make install
  sudo ldconfig

  # === 2) Instala el wrapper Python con el MISMO Python del entorno ===
  cd ../wrappers/python
  conda run --no-capture-output -n "$ENV_NAME" python -m pip install --upgrade pip setuptools wheel
  conda run --no-capture-output -n "$ENV_NAME" python -m pip install .

  # === 3) Verificación dentro del entorno ===
  conda run --no-capture-output -n "$ENV_NAME" python - <<'PY'
import pyrealsense2 as rs, inspect, subprocess
print("pyrealsense2:", rs.__file__)
try:
    so = inspect.getfile(rs)
    print("ldd:")
    print(subprocess.check_output(["ldd", so]).decode())
except Exception:
    pass
PY

  popd >/dev/null
}


verify_pyrealsense_in_env() {
  echo "Verificando import de pyrealsense2 dentro del entorno $ENV_NAME…"
  run_in_env python - <<'PY'
import pyrealsense2 as rs, sys, inspect, subprocess
print("pyrealsense2 OK ->", getattr(rs, "__file__", "(sin ruta)"))
try:
    so = inspect.getfile(rs)
    print("ldd:")
    print(subprocess.check_output(["ldd", so]).decode())
except Exception:
    pass
PY
}

# ===================== MAIN =============================
case "${1:-}" in
  env|deps)
    # 1) Conda + entorno
    install_miniforge_if_needed
    if ! load_conda; then echo "ERROR: No se pudo cargar conda tras instalar Miniforge."; exit 2; fi
    create_env_common

    # 2) Requirements comunes
    install_requirements_common

    if [[ "$IS_JETSON" -eq 1 ]]; then
      # 3) Jetson: compilar librealsense + wrapper en el entorno conda
      install_jetson_system_prereqs
      clone_rules_librealsense
      build_librealsense_cpp_then_wrapper_in_env
      verify_pyrealsense_in_env
      echo "Listo: RealSense (lib + pyrealsense2) instalado para Python $PYTHON_VER en entorno $ENV_NAME (Jetson)."
    else
      # 3) PC: intentar wheel pyrealsense2 dentro del entorno
      install_realsense_pc_in_env
      echo "Listo: Entorno $ENV_NAME con Python $PYTHON_VER preparado en PC."
    fi
    ;;

  check)
    if ! load_conda; then echo "ERROR: No se encontró conda."; exit 2; fi
    run_in_env python - <<'PY'
import sys, platform
print("Python:", sys.version.split()[0], "| Platform:", platform.machine())
try:
    import pyrealsense2 as rs
    print("pyrealsense2:", getattr(rs, "__file__", "(no instalado)"))
except Exception as e:
    print("pyrealsense2: (no importable) ->", e)
PY
    ;;

  realsense-test)
    echo "Iniciando prueba de cámara con pyrealsense2 dentro del entorno $ENV_NAME…"
    if [[ -f "src/utilities/viewCamera.py" ]]; then
      run_in_env python src/utilities/viewCamera.py
    else
      echo "ERROR: No se encontró src/utilities/viewCamera.py"
      exit 1
    fi
    ;;

  visual)
    echo "Placeholder. Usa 'realsense-test' para ejecutar tu visor."
    ;;

  *)
    echo "Uso: $0 {env|deps|check|realsense-test|visual}"
    ;;

esac
