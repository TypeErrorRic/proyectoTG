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

# Compila librealsense C++ + bindings Python e instala el módulo en el env conda (Py 3.8)
build_librealsense_with_bindings_top_cmake_in_env() {
  echo "Compilando librealsense ${RS_TAG} + bindings Python para el entorno '$ENV_NAME'…"

  # 1) Asegura repo + reglas udev
  [[ -d librealsense ]] || git clone https://github.com/IntelRealSense/librealsense.git
  sudo cp librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/ || true
  sudo udevadm control --reload-rules && sudo udevadm trigger

  # 2) Rutas del Python del entorno conda (usa directamente ese intérprete)
  local PY PREFIX PY_INC PY_SITE PY_LIB
  PY="$(conda run -n "$ENV_NAME" which python)"
  PREFIX="$("$PY" - <<'PY'
import sys; print(sys.prefix)
PY
)"
  PY_INC="$("$PY" - <<'PY'
import sysconfig; print(sysconfig.get_paths()["include"])
PY
)"
  PY_SITE="$("$PY" - <<'PY'
import site, sys
# toma el site-packages del propio env (prefijo del env)
candidates = [p for p in site.getsitepackages() if p.startswith(sys.prefix)]
print(candidates[0] if candidates else site.getsitepackages()[0])
PY
)"

  # Intentar localizar la librería de Python
  PY_LIB="$("$PY" - <<'PY'
import sys, glob, os
ver=f"{sys.version_info.major}.{sys.version_info.minor}"
cands=[]
for d in (os.path.join(sys.prefix,"lib"), "/usr/lib/aarch64-linux-gnu", "/usr/local/lib"):
    cands += glob.glob(os.path.join(d, f"libpython{ver}*.so"))
    cands += glob.glob(os.path.join(d, f"libpython{ver}*.so.*"))
print(cands[0] if cands else "")
PY
)"

  # Si no hay lib directa, crea symlink a .so.1.0 dentro del env
  if [[ -z "$PY_LIB" && -d "$PREFIX/lib" ]]; then
    if ls "$PREFIX/lib"/libpython3.8.so.* >/dev/null 2>&1; then
      echo "Creando symlink libpython3.8.so en $PREFIX/lib…"
      ln -sf "$(ls "$PREFIX/lib"/libpython3.8.so.* | head -n1)" "$PREFIX/lib/libpython3.8.so"
      PY_LIB="$PREFIX/lib/libpython3.8.so"
    fi
  fi

  echo "PY      = $PY"
  echo "PREFIX  = $PREFIX"
  echo "PY_INC  = $PY_INC"
  echo "PY_SITE = $PY_SITE"
  echo "PY_LIB  = ${PY_LIB:-<no encontrado>}"

  if [[ -z "$PY_INC" ]]; then
    echo "ERROR: No se pudo resolver PY_INC (headers de Python)."
    return 2
  fi
  if [[ -z "$PY_LIB" ]]; then
    echo "ADVERTENCIA: No se encontró libpython*.so. Intentaré continuar sin -DPYTHON_LIBRARY."
  fi

  # 3) Build + install (CMake top-level instala directo en el site-packages del env)
  pushd librealsense >/dev/null
  git fetch --tags || true
  git checkout "${RS_TAG}"

  rm -rf build && mkdir build && cd build

  # Ayuda a CMake a encontrar cosas del env
  export CMAKE_PREFIX_PATH="${PREFIX}:${CMAKE_PREFIX_PATH:-}"

  # Prepara args CMake
  CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DFORCE_RSUSB_BACKEND=ON
    -DBUILD_PYTHON_BINDINGS=ON
    -DPYTHON_EXECUTABLE="$PY"
    -DPYTHON_INCLUDE_DIR="$PY_INC"
    -DPYTHON_INSTALL_DIR="$PY_SITE"
    -DCMAKE_INSTALL_PREFIX="$PREFIX"
    -DBUILD_EXAMPLES=OFF
    -DBUILD_GRAPHICAL_EXAMPLES=OFF
    -Dpybind11_FINDPYTHON=ON
  )
  # Solo si tenemos la lib, pásala
  if [[ -n "$PY_LIB" ]]; then
    CMAKE_ARGS+=(-DPYTHON_LIBRARY="$PY_LIB")
  fi

  cmake .. "${CMAKE_ARGS[@]}" | tee cmake_config.log

  make -j2 || make -j1
  sudo make install
  sudo ldconfig

  # 4) Verificación dentro del entorno
  conda run --no-capture-output -n "$ENV_NAME" python - <<'PY'
import pyrealsense2 as rs, inspect, subprocess
print("pyrealsense2 OK ->", getattr(rs, "__file__", "(sin ruta)"))
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
      build_librealsense_with_bindings_top_cmake_in_env
      verify_pyrealsense_in_env
      echo "Listo: RealSense (lib + pyrealsense2) instalado para Python $PYTHON_VER en entorno $ENV_NAME (Jetson)."
    else
      # 3) PC: intentar wheel pyrealsense2 dentro del entorno
      install_realsense_pc_in_env
      echo "Listo: Entorno $ENV_NAME con Python $PYTHON_VER preparado en PC."
    fi
    ;;

    check)
    if ! load_conda; then
        echo "ERROR: No se encontró conda."
        exit 2
    fi

    # Exporta IS_JETSON para que el Python interno lo pueda leer
    export IS_JETSON

    echo "==> Verificando pyrealsense2 y paquetes de requirements.txt..."
    run_in_env python - <<'PY'
import os, sys, importlib, subprocess, textwrap

print(f"Python: {sys.version.split()[0]}")

# ---------- Utilidades ----------
SPECIAL = {
    "opencv-python": "cv2",
    "opencv-python-headless": "cv2",
    "pillow": "PIL",
    "scikit-learn": "sklearn",
    "pyyaml": "yaml",
    "python-dateutil": "dateutil",
    # CuPy wheels importan como 'cupy'
    "cupy": "cupy",
    "cupy-cuda102": "cupy",      # <-- agrega esta línea
    "cupy-cuda11x": "cupy",
    "cupy-cuda12x": "cupy",
}

def clean_spec(line: str):
    """
    Devuelve la especificación 'instalable' (con pins/extras) sin comentarios ni marcadores PEP 508.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    # quitar comentarios
    line = line.split("#", 1)[0].strip()
    if not line:
        return None
    # quitar marcadores de entorno ;python_version<"3.9"
    line = line.split(";", 1)[0].strip()
    return line or None

def base_from_spec(spec: str):
    """
    Extrae el nombre base del paquete (sin versión ni extras), para mapear a nombre de módulo.
    """
    cut_chars = set("<>=!~ [")
    out = []
    for ch in spec:
        if ch in cut_chars:
            break
        out.append(ch)
    return "".join(out).strip()

def mod_from_pkgname(pkgname: str):
    return SPECIAL.get(pkgname.lower(), pkgname.replace("-", "_"))

# ---------- pyrealsense2 ----------
rs_ok = False
try:
    import pyrealsense2  # noqa: F401
    print("✅ pyrealsense2: OK")
    rs_ok = True
except Exception as e:
    print(f"❌ pyrealsense2: {e!s}")

# En PC intentamos instalar wheel; en Jetson solo avisamos
is_jetson = os.getenv("IS_JETSON", "0") == "1"
if not rs_ok and not is_jetson:
    print("→ Intentando instalar pyrealsense2 (wheel, PC)…")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyrealsense2"])
        import pyrealsense2  # reintento
        print("✅ pyrealsense2 instalado (wheel)")
        rs_ok = True
    except Exception as e:
        print(f"⚠️  No se pudo instalar pyrealsense2 por wheel: {e!s}")
        print("    Puedes compilarlo con 'env' como en Jetson si lo necesitas en PC.")

if not rs_ok and is_jetson:
    print("ℹ️ En Jetson, instala pyrealsense2 compilando con 'env' (librealsense + bindings).")

# ---------- requirements.txt ----------
req_file = "requirements.txt"
if not os.path.exists(req_file):
    print("⚠️  No se encontró requirements.txt en el directorio actual.")
    sys.exit(0)

print("\n📦 Revisando paquetes de requirements.txt...\n")

missing_specs = []    # lista de líneas tal como deben instalarse con pip
checked = 0

with open(req_file, "r", encoding="utf-8") as f:
    for raw in f:
        spec = clean_spec(raw)
        if not spec:
            continue
        pkg = base_from_spec(spec)
        if not pkg:
            continue
        mod = mod_from_pkgname(pkg)
        try:
            importlib.import_module(mod)
            print(f"✅ {spec}")
        except Exception:
            print(f"❌ {spec}")
            missing_specs.append(spec)
        checked += 1

print(f"\nResumen: {checked} paquetes verificados, {len(missing_specs)} faltantes.")
if missing_specs:
    print("→ Intentando instalar faltantes dentro del entorno activo…")
    failed = []
    for spec in missing_specs:
        print(f"   - pip install {spec}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", spec])
        except Exception as e:
            print(f"     ⚠️  Falló {spec}: {e!s}")
            failed.append(spec)

    if failed:
        print("\n❌ Algunos paquetes no se pudieron instalar automáticamente:")
        for spec in failed:
            print(f"   - {spec}")
        # no salir con error duro para que puedas leer el log completo
    else:
        print("\n✅ Todos los paquetes faltantes fueron instalados correctamente.")

PY
    ;;

  visual)
    if [[ "$IS_JETSON" -eq 1 ]]; then
        echo "No funciona en Jetson."
    else
        # --- Inicializa conda ---
        if ! command -v conda &> /dev/null; then
            echo "Error: conda no está disponible en el PATH"
            exit 1
        fi
        source "$(conda info --base)/etc/profile.d/conda.sh"

        # --- Activa entorno ---
        conda activate TG || { echo "Error al activar entorno TG"; exit 1; }

        # --- Configura entorno Python ---
        export PYTHONUTF8=1
        export PYTHONIOENCODING=utf-8

        # --- Ejecuta visualizador ---
        cd src/data || { echo "No existe src/data"; exit 1; }
        python visualizador.py
    fi
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
  test)
  echo "Iniciando entorno $ENV_NAME…"
    if [[ -f "src/utilities/ransacCellingGround.py" ]]; then
      run_in_env python src/utilities/ransacCellingGround.py
    else
      echo "ERROR: No se encontró src/utilities/ransacCellingGround.py"
      exit 1
    fi
    ;;
  TxInstall)
    # ===================== SOLO JETSON (TX) EN CONDA =====================
    if [[ "$IS_JETSON" -ne 1 ]]; then
      echo "Este target es SOLO para Jetson (transmisor)."
      exit 0
    fi

    echo "==> [Jetson] Instalando paquetes de sistema (GStreamer + OpenCV con soporte GStreamer)..."
    sudo apt-get update
    sudo apt-get install -y \
      python3-opencv \
      gstreamer1.0-tools gstreamer1.0-libav \
      gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
      gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
      nvidia-l4t-gstreamer v4l-utils

    echo "==> [Jetson] Verificando plugins GStreamer (L4T)..."
    set +e
    /usr/bin/gst-inspect-1.0 nvarguscamerasrc >/dev/null 2>&1 && echo "  ✓ nvarguscamerasrc (CSI)" || echo "  ⚠︎ nvarguscamerasrc (no disponible)"
    /usr/bin/gst-inspect-1.0 v4l2src         >/dev/null 2>&1 && echo "  ✓ v4l2src (USB/UVC)"     || echo "  ✗ v4l2src"
    /usr/bin/gst-inspect-1.0 nvvidconv        >/dev/null 2>&1 && echo "  ✓ nvvidconv (NVMM)"      || echo "  ✗ nvvidconv"
    /usr/bin/gst-inspect-1.0 nvv4l2h264enc    >/dev/null 2>&1 && echo "  ✓ nvv4l2h264enc (NVENC)" || echo "  ✗ nvv4l2h264enc"
    /usr/bin/gst-inspect-1.0 h264parse        >/dev/null 2>&1 && echo "  ✓ h264parse"             || echo "  ✗ h264parse"
    /usr/bin/gst-inspect-1.0 rtph264pay       >/dev/null 2>&1 && echo "  ✓ rtph264pay"            || echo "  ✗ rtph264pay"
    set -e

    echo "==> [Jetson] Listo."
    ;;
  run)
    # === Rutas relativas: run.sh -> src -> src/utilities ===
    ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SRC_DIR="${ROOT_DIR}/src"
    UTIL_DIR="${SRC_DIR}/utilities"

    # Usa el TX FIFO (no el appsrc)
    TX="${UTIL_DIR}/tx_fifo.py"
    RX="${UTIL_DIR}/rx_view.py"

    # --- Parámetros estáticos ---
    WIDTH=1280
    HEIGHT=720
    FPS=30
    BITRATE_KBPS=4000
    PORT=5000
    FIFO="/tmp/frames.rgb"   # FIFO compartido entre Python y GStreamer

    # Verificación de archivos
    [[ -f "$TX" ]] || { echo "ERROR: No existe $TX"; exit 2; }
    [[ -f "$RX" ]] || { echo "ERROR: No existe $RX"; exit 2; }

    # Consumir el subcomando para que $1 sea la IP (en Jetson)
    shift

    # ===== Ejecutar SIEMPRE dentro de conda =====
    if [[ "${IS_JETSON:-0}" -eq 1 ]]; then
      # === Jetson -> Transmisor (solo IP como argumento), FIFO + gst-launch ===
      PC_IP="${1:?Uso: $0 link_rgb <PC_IP>}"

      echo "==> Transmisor Jetson (FIFO + gst-launch)"
      echo "IP destino: $PC_IP | ${WIDTH}x${HEIGHT} @ ${FPS} fps | ${BITRATE_KBPS} kbps | puerto ${PORT}"

      # Asegura que exista el FIFO
      if [[ ! -p "$FIFO" ]]; then
        rm -f "$FIFO"
        mkfifo -m 666 "$FIFO" || { echo "ERROR: No se pudo crear FIFO $FIFO"; exit 2; }
      fi

      # Verifica encoder NVENC disponible
      if ! gst-inspect-1.0 nvv4l2h264enc >/dev/null 2>&1; then
        echo "ERROR: No se encontró 'nvv4l2h264enc'. Revisa instalación de GStreamer/NVENC en L4T."
        exit 2
      fi

      echo "==> Lanzando GStreamer (lector del FIFO) ..."
      # Lee RGB crudo del FIFO, convierte a NV12 en NVMM y envía por RTP/UDP (H.264 HW)
      gst-launch-1.0 -v \
        filesrc location="$FIFO" do-timestamp=true ! \
        videoparse width=$WIDTH height=$HEIGHT framerate=$FPS/1 format=rgb ! \
        videoconvert ! \
        nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! \
        nvv4l2h264enc insert-sps-pps=true iframeinterval=$FPS control-rate=1 \
                      bitrate=$((BITRATE_KBPS*1000)) preset-level=2 ! \
        h264parse config-interval=1 ! rtph264pay pt=96 ! \
        udpsink host="$PC_IP" port="$PORT" sync=false \
        2> >(sed -u 's/^/[GST] /' >&2) &
      GST_PID=$!

      # Mata el gst-launch si salimos
      trap 'kill $GST_PID 2>/dev/null || true' EXIT

      echo "==> Ejecutando TX Python (3.8) escribiendo al FIFO..."
      # OJO: aquí NO inyectamos cv2 del sistema; tu Python 3.8 solo escribe bytes al FIFO.
      run_in_env env \
        PYTHONUNBUFFERED=1 \
        PYTHONIOENCODING=utf-8 \
        PYTHONPATH="${SRC_DIR}:${UTIL_DIR}:${PYTHONPATH:-}" \
        python "$TX" \
          --fifo "$FIFO" \
          --width "$WIDTH" --height "$HEIGHT" \
          --fps "$FPS" \
          --pipe-format rgb

      # Espera a que termine gst-launch (si sigue vivo)
      wait $GST_PID || true
      exit 0
    else
      # === PC -> Receptor (Windows, FUERA de conda) ===
      echo "==> Receptor PC (Windows, Python 3.13) escuchando en puerto ${PORT}"

      # Escoge el intérprete: preferimos el launcher de Windows 'py' con 3.13
      if command -v py >/dev/null 2>&1; then
        PY="py"
        PY_VER=("-3.13")
      elif command -v python3 >/dev/null 2>&1; then
        PY="python3"
        PY_VER=()
      elif command -v python >/dev/null 2>&1; then
        PY="python"
        PY_VER=()
      else
        echo "ERROR: No se encontró Python en este PC. Instala Python 3.13 o usa el launcher 'py'."
        exit 2
      fi

      # Si ejecutas desde Git Bash/MSYS, convierte la ruta POSIX a Windows para Python
      RX_PATH="$RX"
      if is_windows_shell && command -v cygpath >/dev/null 2>&1; then
        RX_PATH="$(cygpath -w "$RX")"
      fi

      # Argumentos para el receptor (permite override opcional de GStreamer con GST_PREFIX)
      RX_ARGS=(--port "$PORT")
      if [[ -n "${GST_PREFIX:-}" ]]; then
        RX_ARGS+=(--gst-prefix "$GST_PREFIX")
      fi

      # Ejecuta el receptor
      "$PY" "${PY_VER[@]}" "$RX_PATH" "${RX_ARGS[@]}"
    fi
    ;;
  *)
    echo "Uso: $0 {env|deps|check|realsense-test|visual|run|transmitir}"
    ;;
esac
