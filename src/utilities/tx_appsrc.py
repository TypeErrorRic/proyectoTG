# tx_appsrc.py
#!/usr/bin/env python3
import argparse, time
import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

from viewCamera import init_camera, extract_rgb # <-- tu función en otro archivo

Gst.init(None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="IP del PC receptor")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--bitrate", type=int, default=4000, help="Kbps")
    args = ap.parse_args()

    W, H, FPS = args.width, args.height, args.fps
    BR = args.bitrate * 1000

    # Inicia tu cámara (tu init_camera puede activar depth también; no afecta usar solo extract_rgb)
    pipeline_rs = init_camera(color_width=W, color_height=H, depth_width=W, depth_height=H, fps=FPS)

    # GStreamer: appsrc(BGR) -> videoconvert -> NV12 -> nvvidconv(NVMM) -> nvv4l2h264enc -> RTP -> UDP
    pipe_str = f"""
appsrc name=src is-live=true block=true format=time
  caps=video/x-raw,format=BGR,width={W},height={H},framerate={FPS}/1 !
videoconvert ! video/x-raw,format=NV12 !
nvvidconv ! video/x-raw(memory:NVMM),format=NV12 !
nvv4l2h264enc insert-sps-pps=true bitrate={BR} preset-level=1 iframeinterval={FPS} !
h264parse config-interval=1 ! rtph264pay pt=96 !
udpsink host={args.host} port={args.port}
"""
    gst_pipe = Gst.parse_launch(pipe_str)
    appsrc = gst_pipe.get_by_name("src")
    gst_pipe.set_state(Gst.State.PLAYING)

    duration = Gst.util_uint64_scale_int(1, Gst.SECOND, FPS)
    n = 0
    try:
        while True:
            frames = pipeline_rs.wait_for_frames()
            bgr = extract_rgb(frames)  # <<< SOLO usamos tu función RGB
            if bgr is None:
                continue
            if bgr.shape[1] != W or bgr.shape[0] != H:
                # Ajusta tamaño si la cámara no coincidió exactamente
                import cv2
                bgr = cv2.resize(bgr, (W, H))

            data = bgr.tobytes()
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            buf.pts = buf.dts = n * duration
            buf.duration = duration
            n += 1

            ret = appsrc.emit("push-buffer", buf)
            if ret.value_nicks[ret.value] != 'ok':
                print("push-buffer:", ret)
                break

    except KeyboardInterrupt:
        pass
    finally:
        try:
            appsrc.emit("end-of-stream")
        except Exception:
            pass
        gst_pipe.set_state(Gst.State.NULL)
        try:
            pipeline_rs.stop()
        except Exception:
            pass

if __name__ == "__main__":
    main()