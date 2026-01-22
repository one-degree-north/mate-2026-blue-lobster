import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)

import re
import cairo
import pandas as pd
import numpy as np
import ctypes
import cv2
import os
import shutil
import dearpygui.dearpygui as dpg

dpg.create_context()


dpg.set_global_font_scale(0.5)

dpg.configure_app(docking=True, docking_space=True, init_file="mate.ini")
dpg.create_viewport(title='MATE Client', width=1280, height=720)

with dpg.theme() as theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 4)
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 2)
        dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 2)
        dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 4)
        dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 4, 4)
        dpg.add_theme_style(dpg.mvStyleVar_WindowTitleAlign, 0.5, 0.5)
dpg.bind_theme(theme)

class Photosphere():
    def __init__(self, camera):
        camera.add_callback(self.on_new_frame)
        self.recording = []
        self.is_recording = False
        self.is_writing = False
        self.frame_rate = 30
        self.is_reading = False

        with dpg.window(label="Photosphere"):
            dpg.add_slider_int(label="Frame Rate", default_value=self.frame_rate, min_value=1, max_value=30, callback=self.set_frame_rate)
            self.recording_button = dpg.add_button(label="Start Recording", callback=self.on_recording_button)
            dpg.add_separator()
            self.stitch_button = dpg.add_button(label="Stitch", callback=self.on_stitch_button)
    
    def set_frame_rate(self, _, rate):
        self.frame_rate = rate

    def on_recording_button(self, _):
        if self.is_writing:
            return

        self.is_recording = not self.is_recording
        if not self.is_recording:
            dpg.configure_item(self.recording_button, label="Saving recording...")
            self.is_writing = True

            if os.path.exists("photosphere/recording"):
                shutil.rmtree("photosphere/recording")
            os.makedirs("photosphere/recording", exist_ok=True)

            video_writer = cv2.VideoWriter("photosphere/recording/video.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (1920, 1080))
            for frame in self.recording:
                video_writer.write(frame)
            video_writer.release()

            self.is_writing = False
            dpg.configure_item(self.recording_button, label="Start Recording")


            # for i, frame in enumerate(self.recording):
            #     cv2.imwrite(f"photosphere/recording/frame_{i + 1}.png", frame)
            
            self.is_writing = False
            self.recording = []

            dpg.configure_item(self.recording_button, label="Start Recording")
        else:
            dpg.configure_item(self.recording_button, label="Recording...")
    
    def on_stitch_button(self, _):
        if self.is_reading:
            return
        
        os.makedirs("photosphere", exist_ok=True)

        if os.path.exists("photosphere/stitch"):
            shutil.rmtree("photosphere/stitch")

        os.makedirs("photosphere/stitch", exist_ok=True)

        self.is_reading = True

        dpg.configure_item(self.stitch_button, label="Setting up")
        video_capture = cv2.VideoCapture("photosphere/recording/video.avi")
        # read all frames
        skip_frames = max(1, 30 // self.frame_rate)
        frame_count = 0
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            if frame_count % skip_frames == 0:
                cv2.imwrite(f"photosphere/stitch/frame_{frame_count // skip_frames + 1}.png", frame)
            
            frame_count += 1
        video_capture.release()

        self.is_reading = False
        
        # os.system("/Applications/PTGui.app/Contents/MacOS/PTGui -createproject photosphere/stitch/*.png -output photosphere/stitch/project.pts")
        # os.system("/Applications/PTGui.app/Contents/MacOS/PTGui photosphere/stitch/project.pts")

        dpg.configure_item(self.stitch_button, label = "Stitch")

    def on_new_frame(self, frame):
        if self.is_recording:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            self.recording.append(frame)
        
class CameraStream():
    def __init__(self, port = 5601):
        self.pipeline = Gst.parse_launch(
            # f"udpsrc port={port} ! application/x-rtp,encoding-name=JPEG ! rtpjpegdepay ! jpegdec ! videoconvert ! video/x-raw,format=RGBA ! appsink name=sink"
            f"udpsrc port={port} ! application/x-rtp,encoding-name=H264 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=RGBA ! appsink name=sink"
        )

        self.sink = self.pipeline.get_by_name("sink")
        self.sink.set_property("emit-signals", True)
        self.sink.connect("new-sample", self.on_new_sample, None)
        self.pipeline.set_state(Gst.State.PLAYING)

        self.callbacks = []

        self.stream_dimensions = (640, 480)
        width, height = self.stream_dimensions
        with dpg.texture_registry():
            self.stream_texture_id = dpg.add_raw_texture(
                width=width,
                height=height,
                default_value=np.zeros((height, width, 4), dtype=np.float32),
                format=dpg.mvFormat_Float_rgba
            )
        
    
        self.window = dpg.generate_uuid()    
        with dpg.window(label="Camera Stream", tag=self.window, no_scrollbar=True):
            self.stream_image = dpg.add_image(self.stream_texture_id)

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def update_aspect_ratio(self):
        win_width, win_height = dpg.get_item_rect_size(self.window)
        if win_width == 0 or win_height == 0:
            return

        image_aspect = self.stream_dimensions[0] / self.stream_dimensions[1]
        window_aspect = win_width / win_height

        if window_aspect > image_aspect:
            image_height = win_height
            image_width = int(win_height * image_aspect)
            pos_x = int((win_width - image_width) / 2)
            pos_y = 0
        else:
            image_width = win_width
            image_height = int(win_width / image_aspect)
            pos_x = 0
            pos_y = int((win_height - image_height) / 2)

        dpg.set_item_width(self.stream_image, image_width)
        dpg.set_item_height(self.stream_image, image_height)
        dpg.set_item_pos(self.stream_image, [pos_x, pos_y])
    
    def update(self, frame):
        frame = frame.astype(np.float32) / 255.0

        height, width, _ = frame.shape
        frame_dimensions = (width, height)

        if frame_dimensions != self.stream_dimensions:
            dpg.delete_item(self.stream_texture_id)
            self.stream_dimensions = frame_dimensions
            with dpg.texture_registry():
                self.stream_texture_id = dpg.add_raw_texture(
                    width=width,
                    height=height,
                    default_value=frame,
                    format=dpg.mvFormat_Float_rgba
                )
                dpg.configure_item(self.stream_image, texture_tag=self.stream_texture_id)
        else:
            dpg.set_value(self.stream_texture_id, frame)
        

    def on_new_sample(self, sink, _):
        sample = sink.emit("pull-sample")

        caps = sample.get_caps()
        buffer = sample.get_buffer()

        width = caps.get_structure(0).get_value("width")
        height = caps.get_structure(0).get_value("height")
        
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            print("Failed to map buffer")
            return Gst.FlowReturn.ERROR
        
        frame = np.ndarray(
            (height, width, 4),
            buffer=map_info.data,
            dtype=np.uint8
        )
        self.update(frame)

        for callback in self.callbacks:
            callback(frame)

        buffer.unmap(map_info)

        return Gst.FlowReturn.OK
class Notes():
    def __init__(self):
        self.window = dpg.generate_uuid()
        with dpg.window(label="Notes", tag = self.window):
            self.input = dpg.add_input_text(default_value=
                "Ship Height (Back Right): 30.5 (colored)\n\n\n\n\n\n\n\n\n\nLength: 189",
                multiline=True
            )
    
    def update(self):
        width = dpg.get_item_width(self.window)
        height = dpg.get_item_height(self.window)

        dpg.set_item_width(self.input, width - 16)
        dpg.set_item_height(self.input, height - 40)



camera_stream1 = CameraStream(5000)
#camera_stream2 = CameraStream(5601)
# photogrammetry = Photogrammetry(camera = camera_stream2)
# photosphere = Photosphere(camera = camera_stream2)
# notes = Notes()

dpg.setup_dearpygui()
dpg.show_viewport()
try:
    while dpg.is_dearpygui_running():
        camera_stream1.update_aspect_ratio()
        #camera_stream2.update_aspect_ratio()
        # photogrammetry.update()
        # notes.update()

        dpg.render_dearpygui_frame()
except KeyboardInterrupt:
    pass

dpg.destroy_context()