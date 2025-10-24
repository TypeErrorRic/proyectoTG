#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson TX (sin gi): transmite frames BGR procesados por RTP/UDP (H.264 HW).
Mensajes clave:
  - "[TX] ✅ Tubería GStreamer ABIERTA"  -> el encoder/sink quedó listo
  - "[TX] ✅ Primer frame ENVIADO"       -> transmisión ACTIVA (fluyendo paquetes)

Receptor (PC/Windows) de ejemplo:
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

from viewCamera import init_camera, extract_rgb  # -> BGR uint8 (H,W,3)


def build_gst_pipeline(host: str, port: int, w: int, h: int, fps: int, bitrate_bps: int) -> str:
    return (
        f"videoconvert ! "
        f"video/x-raw,format=NV12,width={w},height={h},framerate={fps}/1 ! "
        f"nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
        f"nvv4l2h264enc insert-sps-pps=true control-rate=1 bitrate={bitrate_bps} "
        f"iframeinterval={fps} preset-level=1 ! "
        f"h264parse ! rtph264pay config-interval=1 pt=96 ! "
        f"udpsink host={host} port={port} sync=false"
    )


def ensure_opencv_gst() -> bool:
    try:
        info = cv2.getBuildInformation()
    except Exception:
        return False
    return ("GStreamer:                     YES" in info) or ("GStreamer: YES" in info)


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

    print(f"[TX] Destino: {args.host}:{args.port} | {W}x{H}@{FPS} | ~{args.bitrate} kbps")

    if not ensure_opencv_gst():
        print(
            "[ERROR] OpenCV sin soporte GStreamer.\n"
            "Instala 'python3-opencv' del sistema o inyecta su ruta en PYTHONPATH para este proceso.",
            file=sys.stderr,
        )
        return 2

    # Inicializa tu cámara / pipeline de captura
    rs_pipeline = init_camera(color_width=W, color_height=H, depth_width=W, depth_height=H, fps=FPS)

    # Crea el VideoWriter hacia la tubería GStreamer
    gst_out = build_gst_pipeline(args.host, args.port, W, H, FPS, BR_BPS)
    out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, FPS, (W, H), True)
    if not out.isOpened():
        print("[ERROR] No se pudo abrir la tubería GStreamer (VideoWriter).", file=sys.stderr)
        print("Pipeline:", gst_out, file=sys.stderr)
        return 3

    print("[TX] ✅ Tubería GStreamer ABIERTA (encoder listo).")
    print("[TX] Enviando RTP/UDP H.264 (NVENC).  Ctrl+C para salir.")

    frame_period = 1.0 / float(FPS)
    t_next = time.perf_counter()
    first_sent = False
    last_beat = time.monotonic()

    try:
        while True:
            frames = rs_pipeline.wait_for_frames()
            bgr = extract_rgb(frames)  # Debe ser BGR uint8 (H,W,3)

            if bgr is None:
                continue
            if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
                print("[WARN] extract_rgb() debe devolver BGR uint8 (H,W,3). Se ignora frame.")
                continue

            if bgr.shape[1] != W or bgr.shape[0] != H:
                bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)

            ok = out.write(bgr)
            if not ok:
                print("[WARN] VideoWriter.write() devolvió False; deteniendo.")
                break

            if not first_sent:
                print("[TX] ✅ Primer frame ENVIADO — transmisión ACTIVA.")
                first_sent = True

            # Heartbeat cada ~5 s (útil para saber que sigue vivo)
            now = time.monotonic()
            if now - last_beat >= 5.0:
                print("[TX] …transmitiendo…")
                last_beat = now

            # Pace simple
            t_next += frame_period
            dt = t_next - time.perf_counter()
            if dt > 0:
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
