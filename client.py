import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)

import ctypes
import cv2
import os
import dearpygui.dearpygui as dpg
import shutil

lib = ctypes.cdll.LoadLibrary("libpgm.dylib")

lib.run_photogrammetry_session.arg_types = [ctypes.c_char_p]
lib.run_photogrammetry_session.restype = None

lib.is_completed.argtypes = []
lib.is_completed.restype = ctypes.c_bool

lib.get_progress.argtypes = []
lib.get_progress.restype = ctypes.c_double

lib.get_eta.argtypes = []
lib.get_eta.restype = ctypes.c_double

lib.stop_photogrammetry_session.argtypes = []
lib.stop_photogrammetry_session.restype = None


class Photogrammetry():

    def set_frame_rate(self, _, frame_rate):
        self.frame_rate = frame_rate

    def on_recording_button(self, _):
        if self.is_writing:
            return
        
        if self.is_recording:
            self.is_recording = False
            os.makedirs("pgm", exist_ok=True)

            if os.path.exists("pgm/recording"):
                shutil.rmtree("pgm/recording")

            os.makedirs("pgm/recording", exist_ok=True)

            dpg.configure_item(self.recording_button, label="Saving Recording")
            self.is_writing = True
            video_writer = cv2.VideoWriter("pgm/recording/video.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (800, 600))
            for frame in self.recording:
                video_writer.write(frame)
            video_writer.release()
            self.is_writing = False

            dpg.configure_item(self.recording_button, label="Start Recording")
        else:
            self.is_recording = True
            dpg.configure_item(self.recording_button, label="Recording...")
    
    def update(self):
        progress = self.lib.get_progress()
        eta = self.lib.get_eta()

        dpg.set_value(self.progress_bar, progress)
        dpg.configure_item(self.progress_bar, overlay=f"{100 * progress:.2f}%")
        dpg.set_value(self.eta_indicator, f"ETA: {eta:.2f}s")

    def toggle_reconstruction(self, _):
        if self.is_reading:
            return

        if self.running:
            self.running = False
            self.lib.stop_photogrammetry_session()
            dpg.configure_item(self.reconstruction_button, label="Start Reconstruction")
        else:
            self.running = True

            os.makedirs("pgm", exist_ok=True)
            if os.path.exists("pgm/reconstruction"):
                shutil.rmtree("pgm/reconstruction")
            os.makedirs("pgm/reconstruction/model", exist_ok=True)

            self.is_reading = True

            dpg.configure_item(self.reconstruction_button, label="Setting up")
            video_capture = cv2.VideoCapture("pgm/recording/video.avi")
            # read all frames
            skip_frames = max(1, 30 // self.frame_rate)
            frame_count = 0
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break

                if frame_count % skip_frames == 0:
                    cv2.imwrite(f"pgm/reconstruction/frame_{frame_count // skip_frames + 1}.png", frame)
                
                frame_count += 1
            video_capture.release()

            self.is_reading = False
                
            # frames = len(os.listdir("pgm/recording"))
            # skip_frames = max(1, 30 // self.frame_rate)
            # for frame in range(1, frames + 1, skip_frames):
            #     shutil.copy(f"pgm/recording/frame_{frame}.png", f"pgm/reconstruction/frame_{frame}.png")

            self.lib.run_photogrammetry_session(b"pgm/reconstruction")
            dpg.configure_item(self.reconstruction_button, label="Stop Reconstruction")

    def __init__(self, camera):
        self.is_recording = False
        self.is_writing = False
        self.is_reading = False
        self.recording = []
        self.running = False
        self.frame_rate = 30

        camera.add_callback(self.on_new_frame)

        with dpg.window(label="Photogrammetry"):
            dpg.add_slider_int(label="Frame Rate", default_value=30, min_value=1, max_value=30, callback=self.set_frame_rate)
            self.recording_button = dpg.add_button(label="Start Recording", callback=self.on_recording_button)
            
            dpg.add_separator()
            dpg.add_spacer()
            
            with dpg.group(horizontal=True):
                self.progress_bar = dpg.add_progress_bar(default_value=0.0)
                self.eta_indicator = dpg.add_text("ETA: 0.0s")

            with dpg.group(horizontal=True):
                self.reconstruction_button = dpg.add_button(label="Start Reconstruction", callback=self.toggle_reconstruction)
                dpg.add_button(label="Open in Preview", callback=lambda: os.system("open pgm/reconstruction/out.usdz"))
                dpg.add_button(label="Open in Meshlab", callback=lambda: os.system("open -a /Applications/MeshLab2023.12.app pgm/reconstruction/model/out.obj"))

    def on_new_frame(self, frame):
        if self.is_recording:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            print("recorded frame")
            self.recording.append(frame)
photogrammetry = Photogrammetry(camera = camera_stream2)

try:
    while dpg.is_dearpygui_running():
        photogrammetry.update()

        dpg.render_dearpygui_frame()
except KeyboardInterrupt:
    pass
