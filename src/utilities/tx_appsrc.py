#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson TX (sin gi): transmite frames BGR procesados por RTP/UDP (H.264 HW).
Receptor (PC/Windows):
  set "GST_PLUGIN_SYSTEM_PATH=C:\gstreamer\1.0\msvc_x86_64\lib\gstreamer-1.0"
  set "PATH=C:\gstreamer\1.0\msvc_x86_64\bin;%PATH%"
  gst-launch-1.0 -v ^
    udpsrc port=5000 caps="application/x-rtp, media=video, encoding-name=H264, payload=96, clock-rate=90000" ! ^
    rtpjitterbuffer latency=100 drop-on-late=true ! ^
    rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
"""

import argparse
import sys
import time
import numpy as np
import cv2

# Debes proveer estas funciones en tu viewCamera.py
from viewCamera import init_camera, extract_rgb  # init_camera(...), extract_rgb(frames) -> np.ndarray BGR (H,W,3)


def build_gst_pipeline(host: str, port: int, w: int, h: int, fps: int, bitrate_bps: int) -> str:
    """
    OpenCV empuja a 'appsrc' implícito. Cadena de salida:
      appsrc -> videoconvert -> NV12 -> nvvidconv(NVMM) ->
      nvv4l2h264enc -> h264parse -> rtph264pay -> udpsink
    """
    return (
        # OpenCV crea appsrc; empezamos con filtros
        f"videoconvert ! "
        f"video/x-raw,format=NV12,width={w},height={h},framerate={fps}/1 ! "
        f"nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
        f"nvv4l2h264enc insert-sps-pps=true control-rate=1 bitrate={bitrate_bps} "
        f"iframeinterval={fps} preset-level=1 ! "
        f"h264parse ! rtph264pay config-interval=1 pt=96 ! "
        f"udpsink host={host} port={port} sync=false"
    )


def ensure_opencv_gst():
    """Comprueba que OpenCV tenga backend GStreamer."""
    try:
        info = cv2.getBuildInformation()
    except Exception:
        return False
    return "GStreamer:                     YES" in info or "GStreamer: YES" in info


def main():
    ap = argparse.ArgumentParser(description="Transmisor Jetson (RTP/UDP H.264) sin gi, con OpenCV+GStreamer.")
    ap.add_argument("--host", required=True, help="IP del PC receptor")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--bitrate", type=int, default=4000, help="Kbps (ej. 4000 = 4 Mbps)")
    args = ap.parse_args()

    W, H, FPS = int(args.width), int(args.height), int(args.fps)
    BR_BPS = int(args.bitrate) * 1000

    print(f"[TX] → {args.host}:{args.port} | {W}x{H}@{FPS} | ~{args.bitrate} kbps")

    if not ensure_opencv_gst():
        print(
            "[ERROR] OpenCV actual no tiene soporte GStreamer.\n"
            "Instala 'python3-opencv' del sistema y/o inyecta su ruta en PYTHONPATH para este proceso.\n"
            "Ejemplo (Jetson, en conda): PYTHONPATH=/usr/lib/python3.8/dist-packages:$PYTHONPATH conda run -n <ENV> python tx_appsrc.py --host ...",
            file=sys.stderr,
        )
        return 2

    # --- Inicializa tu cámara/pipe de adquisición (puede ser RealSense o CSI en tu módulo) ---
    rs_pipeline = init_camera(color_width=W, color_height=H, depth_width=W, depth_height=H, fps=FPS)

    # --- Crea VideoWriter sobre la tubería GStreamer ---
    gst_out = build_gst_pipeline(args.host, args.port, W, H, FPS, BR_BPS)
    out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, FPS, (W, H), True)
    if not out.isOpened():
        print("[ERROR] No se pudo abrir la tubería GStreamer desde OpenCV (VideoWriter).", file=sys.stderr)
        print("Pipeline:", gst_out, file=sys.stderr)
        return 3

    print("[TX] Enviando RTP/UDP H.264 (NVENC).  Ctrl+C para salir.")

    # Ritmo de envío (preciso pero liviano)
    frame_period = 1.0 / float(FPS)
    t_next = time.perf_counter()

    try:
        while True:
            # Obtén tus frames del módulo (puede traer depth también; aquí usamos solo color procesado)
            frames = rs_pipeline.wait_for_frames()
            bgr = extract_rgb(frames)  # Debe devolver BGR uint8 (H,W,3)

            if bgr is None:
                continue
            if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
                print("[WARN] extract_rgb() debe devolver BGR uint8 (H,W,3). Se ignora frame.")
                continue

            # Tamaño exacto para el encoder
            if bgr.shape[1] != W or bgr.shape[0] != H:
                bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)

            # Empuja el frame
            ok = out.write(bgr)
            if not ok:
                print("[WARN] VideoWriter.write() devolvió False; deteniendo.")
                break

            # Pace simple a FPS deseado
            t_next += frame_period
            dt = t_next - time.perf_counter()
            if dt > 0:
                # dormir lo justo; si el procesamiento es pesado, simplemente no dormirá
                time.sleep(dt)

    except KeyboardInterrupt:
        pass
    finally:
        out.release()
        try:
            rs_pipeline.stop()
        except Exception:
            pass

    print("[TX] Fin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())