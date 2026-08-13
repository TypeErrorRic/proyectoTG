#!/usr/bin/env bash
set -euo pipefail

# ================== Configuracion por defecto ==================
PYTHON_BIN="${PYTHON_BIN:-python3.8}"  # Python de sistema (3.8) en Jetson
PYTHON_VER="3.8"

# Tag de librealsense recomendado para JP4.x (estable)
RS_TAG="${RS_TAG:-v2.50.0}"

# ================== Detectores de plataforma ==================
is_cmd() { command -v "$1" >/dev/null 2>&1; }
is_aarch64() { [[ "$(uname -m)" == "aarch64" ]]; }
is_linux() { [[ "${OSTYPE:-}" == linux* ]]; }
is_jetson_board() {
  [[ -f /etc/nv_tegra_release ]] && return 0
  [[ -d /proc/device-tree/tegra-fuse || -d /proc/device-tree/chosen/nvidia,tegra-udrm ]] && return 0
  return 1
}
IS_JETSON=0
if is_aarch64 && is_jetson_board; then IS_JETSON=1; fi

require_jetson() {
  if [[ "$IS_JETSON" -ne 1 ]]; then
    echo "Este script solo soporta Jetson (Linux aarch64)."
    exit 1
  fi
}

ensure_python() {
  if ! is_cmd "$PYTHON_BIN"; then
    echo "ERROR: No se encontro '$PYTHON_BIN' en PATH. Ajusta PYTHON_BIN o instala Python ${PYTHON_VER}."
    exit 2
  fi
  # Asegura que el PYTHONPATH incluya el site-packages donde está pyrealsense2
  export PYTHONPATH="/usr/lib/python3.8/site-packages:/home/jetson/.local/lib/python3.8/site-packages:${PYTHONPATH:-}"
  "$PYTHON_BIN" - <<'PY'
import sys
major, minor = sys.version_info[:2]
if (major, minor) != (3, 8):
    raise SystemExit(f"Python {major}.{minor} detectado, se requiere 3.8")
print(f"Python OK: {sys.version.split()[0]}")
PY
}

# ================== CuPy helpers ===================
setup_cupy_env() {
  # Exporta rutas de includes para NVRTC/CUB cuando usamos CuPy + CUDA10.2 (Jetson).
  # Evita fallos de compilacion JIT que no encuentran util_ptx.cuh.
  local inc cub out
  out="$("$PYTHON_BIN" - <<'PY'
import os, sys
try:
    import cupy
except Exception:
    sys.exit(0)
root = os.path.dirname(cupy.__file__)
inc = os.path.join(root, "core", "include")
cub = os.path.join(inc, "cupy", "cub")
print(inc)
print(cub)
PY
)" || return 0
  # Solo configuramos si logramos resolver rutas
  if [[ -n "$out" ]]; then
    read -r inc cub <<<"$out" || true
    if [[ -n "$inc" ]]; then
      export CUPY_NVRTC_INCLUDE_DIRS="${inc}:${cub}:${CUPY_NVRTC_INCLUDE_DIRS:-}"
      export CUPY_CUB_PATH="${cub}"
    fi
  fi
}

compile_align_ptx_if_missing() {
  # Genera align_depth_to_color.ptx si no existe (para acelerar el arranque de CuPy RawModule).
  local cu="src/application/kernels/align_depth_to_color.cu"
  local ptx="src/application/kernels/align_depth_to_color.ptx"
  [[ -f "$ptx" ]] && return 0
  if ! is_cmd nvcc; then
    echo "Aviso: nvcc no encontrado, se omitira la compilacion del PTX ($ptx)."
    return 0
  fi

  local arch="${ALIGN_PTX_ARCH:-auto}"
  if [[ "$arch" == "auto" ]]; then
    arch="$("$PYTHON_BIN" - <<'PY' || true
try:
    import cupy
    p = cupy.cuda.runtime.getDeviceProperties(0)
    print(f"sm_{p['major']}{p['minor']}")
except Exception:
    pass
PY
)"
  fi
  if [[ -z "${arch:-}" ]]; then
    arch="sm_72"
    echo "Aviso: no se pudo detectar compute capability. Usando valor por defecto ${arch} (ajusta ALIGN_PTX_ARCH si es necesario)."
  fi

  echo "Compilando PTX ${ptx} con arch=${arch}..."
  if ! nvcc -ptx -arch="${arch}" "${cu}" -o "${ptx}"; then
    echo "ADVERTENCIA: fallo compilando PTX (${ptx}). Continua con compilacion JIT en runtime."
  fi
}

# ================== Pausa ===================
pause_if_needed() {
  local want_pause=0
  if [[ "${KEEP_OPEN:-auto}" == "1" ]]; then
    want_pause=1
  fi
  [[ "$want_pause" -eq 0 ]] && return 0
  echo
  if [[ -e /dev/tty ]]; then
    read -rp "Presiona Enter para cerrar..." _ </dev/tty
  elif [[ -t 0 ]]; then
    read -rp "Presiona Enter para cerrar..." _
  else
    echo "Presiona Ctrl+C para cerrar... (cerrando automaticamente en 120s)"
    sleep 120 || true
  fi
}
trap pause_if_needed EXIT

# =================== Jetson (aarch64) ===================
install_jetson_system_prereqs() {
  echo "Instalando prerrequisitos del sistema (Jetson)..."
  sudo apt-get update
  # Preferimos headers/lib de la misma version de Python (3.8); si no existe el paquete, caemos a python3-dev.
  local py_dev_pkg="python${PYTHON_VER}-dev"
  if apt-cache show "$py_dev_pkg" >/dev/null 2>&1; then
    sudo apt-get install -y python3-pip "$py_dev_pkg"
  else
    echo "No se encontro $py_dev_pkg, intentando con python3-dev (puede ser 3.6 en JP4.x)."
    sudo apt-get install -y python3-pip python3-dev
  fi
  sudo apt-get install -y libssl-dev libusb-1.0-0-dev pkg-config \
                          libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
                          cmake build-essential git udev
}

install_requirements_common() {
  [[ -f requirements.txt ]] || { echo "ADVERTENCIA: no hay requirements.txt."; return 0; }
  echo "Instalando dependencias (pip) con $PYTHON_BIN..."
  "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
  "$PYTHON_BIN" -m pip install -r requirements.txt
}

clone_rules_librealsense() {
  [[ -d librealsense ]] || git clone https://github.com/IntelRealSense/librealsense.git
  sudo cp librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/ || true
  sudo udevadm control --reload-rules && sudo udevadm trigger
}

build_librealsense_with_bindings() {
  echo "Compilando librealsense ${RS_TAG} + bindings Python (${PYTHON_BIN})..."

  local PY PREFIX PY_INC PY_SITE PY_LIB
  # CMake/pybind11 necesita una ruta absoluta. Si recibe solo "python3.8",
  # FindPythonLibsNew la interpreta relativa a librealsense/build.
  PY="$(command -v "$PYTHON_BIN")"
  if [[ -z "$PY" || ! -x "$PY" ]]; then
    echo "ERROR: No se pudo resolver el ejecutable '$PYTHON_BIN'."
    return 2
  fi
  PREFIX="$("$PY" - <<'PY'
import sys; print(sys.prefix)
PY
)"
  PY_INC="$("$PY" - <<'PY'
import sysconfig; print(sysconfig.get_paths()["include"])
PY
)"
  PY_SITE="$("$PY" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"

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

  if [[ -z "$PY_LIB" && -d "$PREFIX/lib" ]]; then
    if ls "$PREFIX/lib"/libpython3.8.so.* >/dev/null 2>&1; then
      echo "Creando symlink libpython3.8.so en $PREFIX/lib..."
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
    echo "ADVERTENCIA: No se encontro libpython*.so. Continuando sin -DPYTHON_LIBRARY."
  fi

  pushd librealsense >/dev/null
  git fetch --tags || true
  git checkout "${RS_TAG}"

  rm -rf build && mkdir build && cd build

  export CMAKE_PREFIX_PATH="${PREFIX}:${CMAKE_PREFIX_PATH:-}"

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
  if [[ -n "$PY_LIB" ]]; then
    CMAKE_ARGS+=(-DPYTHON_LIBRARY="$PY_LIB")
  fi

  cmake .. "${CMAKE_ARGS[@]}" | tee cmake_config.log
  make -j2 || make -j1
  sudo make install
  sudo ldconfig

  export PYTHONPATH="/usr/lib/python3.8/site-packages:/home/jetson/.local/lib/python3.8/site-packages:${PYTHONPATH:-}"
  "$PYTHON_BIN" - <<'PY'
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

verify_pyrealsense() {
  echo "Verificando import de pyrealsense2 con $PYTHON_BIN..."
  "$PYTHON_BIN" - <<'PY'
import pyrealsense2 as rs, inspect, subprocess
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
    require_jetson
    install_jetson_system_prereqs
    ensure_python
    install_requirements_common
    clone_rules_librealsense
    build_librealsense_with_bindings
    verify_pyrealsense
    echo "Listo: RealSense (lib + pyrealsense2) instalado para Python $PYTHON_VER en Jetson (sin conda)."
    ;;

  check)
    require_jetson
    ensure_python

    # Asegura que el PYTHONPATH incluya el site-packages donde está pyrealsense2
    export PYTHONPATH="/usr/lib/python3.8/site-packages:/home/jetson/.local/lib/python3.8/site-packages:${PYTHONPATH:-}"

    echo "==> Verificando pyrealsense2 y paquetes de requirements.txt..."
    "$PYTHON_BIN" - <<'PY'
import os, sys, importlib, subprocess

print(f"Python: {sys.version.split()[0]}")

SPECIAL = {
    "opencv-python": "cv2",
    "opencv-python-headless": "cv2",
    "pillow": "PIL",
    "scikit-learn": "sklearn",
    "pyyaml": "yaml",
    "python-dateutil": "dateutil",
    "cupy": "cupy",
    "cupy-cuda102": "cupy",
    "cupy-cuda11x": "cupy",
    "cupy-cuda12x": "cupy",
}

def clean_spec(line: str):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    line = line.split("#", 1)[0].strip()
    if not line:
        return None
    line = line.split(";", 1)[0].strip()
    return line or None

def base_from_spec(spec: str):
    cut_chars = set("<>=!~ [")
    out = []
    for ch in spec:
        if ch in cut_chars:
            break
        out.append(ch)
    return "".join(out).strip()

def mod_from_pkgname(pkgname: str):
    return SPECIAL.get(pkgname.lower(), pkgname.replace("-", "_"))

try:
    import pyrealsense2  # noqa: F401
    print("OK pyrealsense2")
except Exception as e:
    print(f"!! pyrealsense2: {e}")
    print("Compila con './run20.sh env' para instalar pyrealsense2 en Jetson.")

req_file = "requirements.txt"
if not os.path.exists(req_file):
    print("Aviso: no se encontro requirements.txt en el directorio actual.")
    sys.exit(0)

print("\nRevisando paquetes de requirements.txt...\n")

missing_specs = []
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
            print(f"OK {spec}")
        except Exception:
            print(f"!! {spec}")
            missing_specs.append(spec)
        checked += 1

print(f"\nResumen: {checked} paquetes verificados, {len(missing_specs)} faltantes.")
if missing_specs:
    print("Instalando faltantes dentro del Python activo...")
    failed = []
    for spec in missing_specs:
        print(f"   - pip install {spec}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", spec])
        except Exception as e:
            print(f"     !! Fallo {spec}: {e!s}")
            failed.append(spec)

    if failed:
        print("\nAlgunos paquetes no se pudieron instalar automaticamente:")
        for spec in failed:
            print(f"   - {spec}")
    else:
        print("\nTodos los paquetes faltantes fueron instalados correctamente.")
PY
    setup_cupy_env
    ;;

  realsense-test)
    require_jetson
    ensure_python
    setup_cupy_env
    compile_align_ptx_if_missing
    echo "Iniciando prueba de camara con pyrealsense2..."
    if [[ -f "src/application/helpers/camara.py" ]]; then
      export PYTHONPATH="/usr/lib/python3.8/site-packages:/home/jetson/.local/lib/python3.8/site-packages:${PYTHONPATH:-}"
      "$PYTHON_BIN" src/application/helpers/camara.py
    else
      echo "ERROR: No se encontro src/application/helpers/camara.py"
      exit 1
    fi
    ;;
  test)
    require_jetson
    ensure_python
    setup_cupy_env
    compile_align_ptx_if_missing
    echo "Iniciando main..."
    if [[ -f "src/main.py" ]]; then
      export PYTHONPATH="/usr/lib/python3.8/site-packages:/home/jetson/.local/lib/python3.8/site-packages:${PYTHONPATH:-}"
      "$PYTHON_BIN" src/main.py
    else
      echo "ERROR: No se encontro src/main.py"
      exit 1
    fi
    ;;
  test-2)
    require_jetson
    ensure_python
    setup_cupy_env
    compile_align_ptx_if_missing
    echo "Iniciando prueba de camara (test-2)..."
    if [[ -f "src/application/helpers/camara.py" ]]; then
      export PYTHONPATH="/usr/lib/python3.8/site-packages:/home/jetson/.local/lib/python3.8/site-packages:${PYTHONPATH:-}"
      "$PYTHON_BIN" src/application/helpers/camara.py
    else
      echo "ERROR: No se encontro src/application/helpers/camara.py"
      exit 1
    fi
    ;;
  rgb-depth)
    require_jetson
    ensure_python
    setup_cupy_env
    compile_align_ptx_if_missing
    echo "Iniciando visor RGB + Depth..."
    if [[ -f "src/infrastructure/datasets/RGB_DEPTH.py" ]]; then
      export PYTHONPATH="/usr/lib/python3.8/site-packages:/home/jetson/.local/lib/python3.8/site-packages:${PYTHONPATH:-}"
      "$PYTHON_BIN" src/infrastructure/datasets/RGB_DEPTH.py
    else
      echo "ERROR: No se encontro src/infrastructure/datasets/RGB_DEPTH.py"
      exit 1
    fi
    ;;
  eval-nyu)
    require_jetson
    ensure_python
    setup_cupy_env
    echo "Evaluando NYU V2 con AlgoritmosSegmentacion..."
    if [[ -f "tests/evaluation/evaluate_nyu_v2.py" ]]; then
      export PYTHONPATH="/usr/lib/python3.8/site-packages:/home/jetson/.local/lib/python3.8/site-packages:${PYTHONPATH:-}"
      "$PYTHON_BIN" tests/evaluation/evaluate_nyu_v2.py "${@:2}"
    else
      echo "ERROR: No se encontro tests/evaluation/evaluate_nyu_v2.py"
      exit 1
    fi
    ;;
  resultadosTG)
    require_jetson
    ensure_python
    setup_cupy_env
    compile_align_ptx_if_missing
    echo "Procesando videos RGB-D grabados y guardando RGB + overlay..."
    if [[ -f "tests/video/scripts/extractVideoFrames.py" ]]; then
      export PYTHONPATH="/usr/lib/python3.8/site-packages:/home/jetson/.local/lib/python3.8/site-packages:${PYTHONPATH:-}"
      PYTHONFAULTHANDLER=1 "$PYTHON_BIN" tests/video/scripts/extractVideoFrames.py "${@:2}"
    else
      echo "ERROR: No se encontro tests/video/scripts/extractVideoFrames.py"
      exit 1
    fi
    ;;
  comparar-anotaciones)
    require_jetson
    ensure_python
    echo "Extrayendo anotaciones, comparando mascaras y generando el reporte JSON..."
    if [[ -f "tests/video/scripts/extractAnnotationMasks.py" ]]; then
      export PYTHONPATH="/usr/lib/python3.8/site-packages:/home/jetson/.local/lib/python3.8/site-packages:${PYTHONPATH:-}"
      "$PYTHON_BIN" tests/video/scripts/extractAnnotationMasks.py "${@:2}"
    else
      echo "ERROR: No se encontro tests/video/scripts/extractAnnotationMasks.py"
      exit 1
    fi
    ;;
  metrics)
    require_jetson
    ensure_python
    setup_cupy_env
    echo "Evaluando metricas con AlgoritmosSegmentacion (dataset PNG en tests/data)..."
    if [[ -f "tests/evaluation/metrics.py" ]]; then
      export PYTHONPATH="/usr/lib/python3.8/site-packages:/home/jetson/.local/lib/python3.8/site-packages:${PYTHONPATH:-}"
      "$PYTHON_BIN" tests/evaluation/metrics.py "${@:2}"
    else
      echo "ERROR: No se encontro tests/evaluation/metrics.py"
      exit 1
    fi
    ;;
  build-engine)
    require_jetson
    ensure_python
    echo "Convirtiendo ONNX a TensorRT engine..."
    echo "Este proceso puede tardar 5-15 minutos en Jetson Nano. Por favor, espera..."
    if [[ -f "src/infrastructure/inference/onnx_to_engine.py" ]]; then
      export PYTHONPATH="/usr/lib/python3.8/site-packages:/home/jetson/.local/lib/python3.8/site-packages:${PYTHONPATH:-}"
      "$PYTHON_BIN" src/infrastructure/inference/onnx_to_engine.py
    else
      echo "ERROR: No se encontro src/infrastructure/inference/onnx_to_engine.py"
      exit 1
    fi
    ;;
  test-trt)
    require_jetson
    ensure_python
    echo "Iniciando prueba de inferencia TensorRT..."
    if [[ -f "src/infrastructure/inference/trt_inference.py" ]]; then
      if [[ ! -f "src/infrastructure/inference/doors/bisenetv2.engine" ]]; then
        echo "ERROR: No se encontro el archivo .engine"
        echo "Ejecuta primero: ./run20.sh build-engine"
        exit 1
      fi
      export PYTHONPATH="/usr/lib/python3.8/site-packages:/home/jetson/.local/lib/python3.8/site-packages:${PYTHONPATH:-}"
      "$PYTHON_BIN" src/infrastructure/inference/trt_inference.py
    else
      echo "ERROR: No se encontro src/infrastructure/inference/trt_inference.py"
      exit 1
    fi
    ;;
  *)
    echo "Uso: $0 {env|deps|check|realsense-test|test|test-2|rgb-depth|eval-nyu|resultadosTG|comparar-anotaciones|metrics|build-engine|test-trt}"
    ;;
esac
