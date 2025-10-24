#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Receptor PC (Windows) sin gi:
Lanza gst-launch-1.0 para recibir RTP/UDP H.264 y mostrar video.

Ejemplos:
  py -3.13 rx_view.py                 # puerto 5000, decodificación por software
  py -3.13 rx_view.py --port 6000     # cambiar puerto
  py -3.13 rx_view.py --hw            # decodificación por hardware (d3d11h264dec)
  py -3.13 rx_view.py --gst-prefix "C:\\Program Files\\gstreamer\\1.0\\msvc_x86_64"
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict

# Ruta por defecto (instalador oficial MSVC x86_64)
DEFAULT_GST_PREFIX = r"C:\Program Files\gstreamer\1.0\msvc_x86_64"


def _has_plugin(bin_dir: Path, env: Dict[str, str], plugin: str) -> bool:
    exe = str(bin_dir / "gst-inspect-1.0.exe")
    try:
        r = subprocess.run([exe, plugin], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        return r.returncode == 0
    except Exception:
        return False


def find_gst_launch(user_prefix: Optional[str] = None) -> Tuple[str, dict]:
    """
    Busca gst-launch-1.0 priorizando instalaciones completas (con 'udpsrc'):

      1) --gst-prefix (si se pasó)
      2) DEFAULT_GST_PREFIX (Program Files)
      3) Lo que haya en PATH (solo si contiene 'udpsrc')

    Devuelve (ruta_gst_launch, env_modificado).
    """
    env0 = os.environ.copy()

    # Candidatos preferidos
    candidates = []
    if user_prefix:
        candidates.append(user_prefix)
    candidates.append(DEFAULT_GST_PREFIX)

    for prefix in candidates:
        pfx = Path(prefix)
        bin_dir = pfx / "bin"
        gst_launch = bin_dir / "gst-launch-1.0.exe"
        plug_dir = pfx / "lib" / "gstreamer-1.0"
        if gst_launch.exists():
            env = env0.copy()
            env["PATH"] = f"{bin_dir};{env.get('PATH','')}"
            if plug_dir.exists():
                env["GST_PLUGIN_SYSTEM_PATH"] = str(plug_dir)
            if _has_plugin(bin_dir, env, "udpsrc"):
                return str(gst_launch), env

    # Último recurso: PATH del sistema
    exe = shutil.which("gst-launch-1.0.exe") or shutil.which("gst-launch-1.0")
    if exe:
        bin_dir = Path(exe).parent
        env = env0.copy()
        plug_dir = bin_dir.parent / "lib" / "gstreamer-1.0"
        if plug_dir.exists():
            env["GST_PLUGIN_SYSTEM_PATH"] = str(plug_dir)
        if _has_plugin(bin_dir, env, "udpsrc"):
            return exe, env

    raise FileNotFoundError(
        "No se encontró un GStreamer válido con el plugin 'udpsrc'.\n"
        f"- Instala el oficial y/o usa --gst-prefix \"{DEFAULT_GST_PREFIX}\".\n"
        "- O ajusta PATH/GST_PLUGIN_SYSTEM_PATH para apuntar a esa instalación."
    )


def _choose_decoder(bin_dir: Path, env: Dict[str, str], prefer_hw: bool) -> Tuple[str, str]:
    """
    Devuelve (decoder_segment, modo) en función de plugins disponibles.
    Intenta:
      - d3d11h264dec (HW)
      - avdec_h264 (SW, libav)
      - openh264dec (SW, openh264)
      - decodebin (último recurso)
    """
    if prefer_hw and _has_plugin(bin_dir, env, "d3d11h264dec"):
        return "d3d11h264dec ! d3d11convert ! d3d11videosink sync=false", "HW:d3d11h264dec"
    if _has_plugin(bin_dir, env, "avdec_h264"):
        return "avdec_h264 ! videoconvert ! autovideosink sync=false", "SW:avdec_h264"
    if _has_plugin(bin_dir, env, "openh264dec"):
        return "openh264dec ! videoconvert ! autovideosink sync=false", "SW:openh264dec"
    # Si no hay decodificadores específicos, probar decodebin
    return "decodebin ! videoconvert ! autovideosink sync=false", "auto:decodebin"


def build_pipeline(port: int, decoder_segment: str) -> str:
    """
    Pipeline de recepción:
      udpsrc → rtpjitterbuffer → rtph264depay → h264parse → dec → sink
    Nota: sin 'drop-on-late' por compatibilidad.
    """
    caps = 'application/x-rtp, media=video, encoding-name=H264, payload=96, clock-rate=90000'
    return (
        f'udpsrc port={port} caps="{caps}" ! '
        f'rtpjitterbuffer latency=100 ! '
        f'rtph264depay ! h264parse ! {decoder_segment}'
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Receptor RTP/UDP H.264 (Windows, sin gi).")
    ap.add_argument("--port", type=int, default=5000, help="Puerto UDP a escuchar (default: 5000)")
    ap.add_argument("--hw", action="store_true", help="Decodificación por hardware (d3d11h264dec)")
    ap.add_argument("--gst-prefix", type=str, default=None,
                    help=r"Prefijo de instalación de GStreamer (ej: C:\Program Files\gstreamer\1.0\msvc_x86_64)")
    args = ap.parse_args()

    try:
        gst_launch, env = find_gst_launch(args.gst_prefix)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2

    # Elegir decodificador según plugins detectados
    bin_dir = Path(gst_launch).parent
    decoder_segment, mode = _choose_decoder(bin_dir, env, args.hw)
    pipeline = build_pipeline(args.port, decoder_segment)

    print("[RX] gst-launch:", gst_launch)
    print("[RX] GST_PLUGIN_SYSTEM_PATH:", env.get("GST_PLUGIN_SYSTEM_PATH", "<no definido>"))
    print(f"[RX] Escuchando UDP {args.port}  (Modo decod.: {mode})")
    print("[RX] Pipeline:")
    print("     ", pipeline)
    print("[RX] Nota: si no ves video, revisa el firewall de Windows (permitir UDP entrante en ese puerto).")

    try:
        cmd = f'"{gst_launch}" -v {pipeline}'
        ret = subprocess.call(cmd, shell=True, env=env)
        if ret != 0:
            print(f"[RX] gst-launch terminó con código {ret}", file=sys.stderr)
            return ret
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
