"""
Transmisor de frames crudos (RGB/BGR) desde cámara RealSense hacia FIFO para GStreamer.
Usa gst-launch-1.0 externo para leer del FIFO, codificar H.264 y transmitir por RTP/UDP.
"""

import argparse
import os
import stat
import sys
import time
from typing import Literal

import numpy as np
import cv2

from viewCamera import init_camera, extract_rgb  # Debe entregar BGR uint8 (H,W,3)


def ensure_fifo(path: str, mode: int = 0o666) -> None:
    """Crea el FIFO si no existe. Si existe pero no es FIFO, lanza error."""
    if os.path.exists(path):
        st = os.stat(path)
        if not stat.S_ISFIFO(st.st_mode):
            raise RuntimeError(f"Ruta existe pero NO es FIFO: {path}")
        return
    try:
        os.mkfifo(path, mode)
    except PermissionError:
        # Intento con mkfifo del sistema (por si permisos limitan os.mkfifo)
        rc = os.system(f"mkfifo -m {oct(mode)[2:]} {path}")
        if rc != 0:
            raise


def get_frames(rs_pipeline, W, H, fmt):
    """Obtiene y procesa los fotogramas desde la cámara."""
    frames = rs_pipeline.wait_for_frames()
    bgr = extract_rgb(frames)  # Se asume BGR uint8 (H,W,3)

    if bgr is None:
        return None
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        print("[WARN] extract_rgb() debe devolver BGR uint8 (H,W,3). Se ignora frame.")
        return None

    if bgr.shape[1] != W or bgr.shape[0] != H:
        bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)

    if fmt == "rgb":
        # Convierte a RGB si el videoparse espera RGB
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).tobytes()
    else:
        # Escribe BGR tal cual (y usa format=bgr en videoparse)
        return bgr.tobytes()


def main():
    ap = argparse.ArgumentParser(description="Transmisor Jetson por FIFO (RTP/UDP H.264 via gst-launch externo).")
    ap.add_argument("--fifo", default="/tmp/frames.rgb", help="Ruta del FIFO compartido con gst-launch.")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument(
        "--pipe-format",
        choices=["rgb", "bgr"],
        default="bgr",
        help="Formato de píxel que ESCRIBE este script en el FIFO (videoparse debe coincidir).",
    )
    args = ap.parse_args()

    W, H, FPS = int(args.width), int(args.height), int(args.fps)
    fmt: Literal["rgb", "bgr"] = args.pipe_format

    print(f"[TX] FIFO: {args.fifo} | {W}x{H}@{FPS} | formato FIFO={fmt}")
    # 1) Asegura FIFO
    try:
        ensure_fifo(args.fifo, 0o666)
    except Exception as e:
        print(f"[ERROR] No se pudo crear/verificar FIFO: {e}", file=sys.stderr)
        return 2

    # 2) Inicializa tu cámara/pipeline
    try:
        rs_pipeline = init_camera(
            color_width=W, color_height=H,
            depth_width=W, depth_height=H,
            fps=FPS
        )
    except Exception as e:
        print(f"[ERROR] init_camera() falló: {e}", file=sys.stderr)
        return 3

    # 3) Abre FIFO para escritura (bloquea hasta que gst-launch lo abra en lectura)
    print("[TX] Esperando a que GStreamer abra el FIFO en lectura…")
    try:
        f = open(args.fifo, "wb", buffering=0)  # sin buffer para reducir latencia
    except Exception as e:
        print(f"[ERROR] No se pudo abrir FIFO para escritura: {e}", file=sys.stderr)
        try:
            rs_pipeline.stop()
        except Exception:
            pass
        return 4

    print("[TX] ✅ FIFO ABIERTO PARA ESCRITURA.")
    print("[TX] Escribe frames crudos hacia GStreamer.  Ctrl+C para salir.")

    frame_period = 1.0 / float(FPS)
    t_next = time.perf_counter()
    first_sent = False
    last_beat = time.monotonic()

    try:
        while True:
            buf = get_frames(rs_pipeline, W, H, fmt)
            if buf is None:
                continue

            try:
                f.write(buf)  # EXACTAMENTE H*W*3 bytes por frame
            except BrokenPipeError:
                print("[ERROR] Broken pipe: el proceso GStreamer cerró el FIFO.", file=sys.stderr)
                break
            except Exception as e:
                print(f"[ERROR] Falló la escritura al FIFO: {e}", file=sys.stderr)
                break

            if not first_sent:
                print("[TX] ✅ Primer frame ESCRITO — flujo hacia encoder ACTIVO.")
                first_sent = True

            # Heartbeat cada ~5 s
            now = time.monotonic()
            if now - last_beat >= 5.0:
                print("[TX] …escribiendo al FIFO…")
                last_beat = now

            # Ritmo simple a FPS objetivo
            t_next += frame_period
            dt = t_next - time.perf_counter()
            if dt > 0:
                time.sleep(dt)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            f.close()
        except Exception:
            pass
        try:
            rs_pipeline.stop()
        except Exception:
            pass

    print("[TX] Fin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
