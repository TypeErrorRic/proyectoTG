# rx_view.py
#!/usr/bin/env python3
import argparse, sys, signal, gi
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GLib

Gst.init(None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    pipe_str = (
        f"udpsrc port={args.port} "
        f"caps=application/x-rtp,media=video,encoding-name=H264,payload=96 ! "
        f"rtph264depay ! avdec_h264 ! videoconvert ! autovideosink sync=false"
    )
    pipeline = Gst.parse_launch(pipe_str)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_msg(bus, msg):
        t = msg.type
        if t == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print("ERROR:", err, dbg, file=sys.stderr)
            loop.quit()
        elif t == Gst.MessageType.EOS:
            loop.quit()
    bus.connect("message", on_msg)

    def handle_sigint(*_):
        pipeline.set_state(Gst.State.NULL)
        loop.quit()
    signal.signal(signal.SIGINT, handle_sigint)

    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()
