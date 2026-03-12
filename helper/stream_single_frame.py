import gi, cv2, numpy as np, time
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GObject

Gst.init(None)

# Load image
img = cv2.imread("assets/monkeydo.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape

# Create pipeline
pipeline = Gst.parse_launch(
    f"appsrc name=src ! videoconvert ! video/x-raw,format=I420 ! x264enc tune=zerolatency speed-preset=ultrafast bitrate=5000 key-int-max=30 ! rtph264pay pt=96 config-interval=1 ! udpsink host=127.0.0.1 port=5000"
)

appsrc = pipeline.get_by_name("src")
appsrc.set_property("caps", Gst.Caps.from_string(f"video/x-raw,format=RGB,width={w},height={h},framerate=30/1"))
appsrc.set_property("is-live", True)
appsrc.set_property("block", True)
appsrc.set_property("format", Gst.Format.TIME)


pipeline.set_state(Gst.State.PLAYING)

try:
    while True:
        buf = Gst.Buffer.new_allocate(None, img.nbytes, None)
        buf.fill(0, img.tobytes())
        buf.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, 30)  # frame duration
        ret = appsrc.emit("push-buffer", buf)
        time.sleep(1/30)
except KeyboardInterrupt:
    pipeline.set_state(Gst.State.NULL)
