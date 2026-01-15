import cv2
import numpy as np
import gi
import sys
from nicegui import ui, app
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Initialize GStreamer
Gst.init(None)

# Example pipeline
# This can be replaced with your camera, RTSP, or video file
pipeline_str =  f"""
udpsrc port=5000 caps="application/x-rtp, media=video, encoding-name=H264, payload=96" ! \
rtph264depay ! avdec_h264 ! videoconvert ! video/x-raw,format=RGB ! \
appsink name=sink emit-signals=true max-buffers=1 drop=true
"""

# Create pipeline
pipeline = Gst.parse_launch(pipeline_str)
appsink = pipeline.get_by_name("sink")

# Start pipeline
pipeline.set_state(Gst.State.PLAYING)


img = ui.image()


def update_frame():
    sample = appsink.emit("pull-sample")
    if sample:
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        # Extract width and height from caps
        structure = caps.get_structure(0)
        width = structure.get_value('width')
        height = structure.get_value('height')

        # Get numpy array from Gst.Buffer
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return

        frame = np.frombuffer(map_info.data, dtype=np.uint8)
        frame = frame.reshape((height, width, 3))
        buffer.unmap(map_info)

        _, jpeg_bytes = cv2.imencode('.jpg', frame)
        img.update(jpeg_bytes.tobytes())

        # Process with OpenCV
        cv2.imshow("Frame", frame)

def handle_shutdown():
    pipeline.set_state(Gst.State.NULL)
    cv2.destroyAllWindows()
    sys.exit()

app.on_shutdown(handle_shutdown)
ui.timer(1/30, update_frame)
ui.run(native=True)