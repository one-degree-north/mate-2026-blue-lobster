import OpenGL.GL as gl
import numpy as np
import gi
import threading
import queue
import datetime


gi.require_version("Gst", "1.0")
from gi.repository import Gst
class VideoStream:
    def __init__(self, pipeline_desc, recognizer):
        Gst.init(None)

        self.pipeline = Gst.parse_launch(pipeline_desc)
        self.appsink = self.pipeline.get_by_name("sink")
        self.tee = self.pipeline.get_by_name("t")

        self.raw_frame = None
        self.processed_frame = None

        self.frame_queue = queue.Queue(maxsize=1)
        self.lock = threading.Lock()

        self.recognizer = recognizer

        self.tex_raw = None
        self.tex_processed = None
        self.texture_sizes = {}

        self.appsink.connect("new-sample", self._on_new_sample)
        self.pipeline.set_state(Gst.State.PLAYING)

        threading.Thread(target=self._cv_worker, daemon=True).start()

    # ---------- GStreamer ----------
    def _on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        buffer = sample.get_buffer()
        caps = sample.get_caps()

        width = caps.get_structure(0).get_value("width")
        height = caps.get_structure(0).get_value("height")

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK

        frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape(
            (height, width, 3)
        ).copy()

        buffer.unmap(map_info)

        if not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass

        self.frame_queue.put(frame)
        return Gst.FlowReturn.OK

    # ---------- CV ----------
    def _cv_worker(self):
        while True:
            frame = self.frame_queue.get()

            processed = self.recognizer.outline_object(frame.copy())

            if processed.shape != frame.shape:
                processed = frame

            with self.lock:
                self.raw_frame = frame
                self.processed_frame = processed

    # ---------- OpenGL ----------
    def _update_texture(self, tex, frame):
        if frame is None:
            return tex

        h, w, c = frame.shape
        if c != 3:
            return tex

        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)

        if tex is None or self.texture_sizes.get(tex) != (w, h):
            if tex:
                gl.glDeleteTextures([tex])

            tex = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0,
                gl.GL_RGB,
                w,
                h,
                0,
                gl.GL_RGB,
                gl.GL_UNSIGNED_BYTE,
                None,
            )

            self.texture_sizes[tex] = (w, h)

        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            w,
            h,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            frame,
        )

        return tex

    # ---------- Public API ----------
    def update(self):
        """Call once per frame from render loop"""
        with self.lock:
            self.tex_raw = self._update_texture(self.tex_raw, self.raw_frame)
            self.tex_processed = self._update_texture(
                self.tex_processed, self.processed_frame
            )

    def get_textures(self):
        return self.tex_raw, self.tex_processed

    def get_frame_shape(self):
        if self.raw_frame is None:
            return None
        h, w, _ = self.raw_frame.shape
        return h, w

    def shutdown(self):
        self.pipeline.set_state(Gst.State.NULL)
        if self.tex_raw:
            gl.glDeleteTextures([self.tex_raw])
        if self.tex_processed:
            gl.glDeleteTextures([self.tex_processed])
    

    def start_recording(self):
        # Create a new unique filename
        filename = f"recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        # Create recording bin
        record_bin = Gst.parse_bin_from_description(
            f"queue ! x264enc ! mp4mux ! filesink location={filename}", True
        )
        
        # Add the bin to the pipeline
        pipeline.add(record_bin)
        record_bin.sync_state_with_parent()
        
        # Request a tee pad and link
        tee_src_pad = tee.get_request_pad("src_%u")
        bin_sink_pad = record_bin.get_static_pad("sink")
        tee_src_pad.link(bin_sink_pad)
        
        print(f"Recording started: {filename}")
        return record_bin, tee_src_pad

