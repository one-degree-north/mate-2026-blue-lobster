import ctypes
import cv2
import os
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

    def start_recording(self):
        if self.is_writing:
            return
        self.recording = []
        self.is_recording = True

    def stop_recording(self):
        if not self.is_recording:
            return
        if self.is_writing:
            return
        
        self.is_recording = False
        os.makedirs("pgm", exist_ok=True)

        if os.path.exists("pgm/recording"):
            shutil.rmtree("pgm/recording")

        os.makedirs("pgm/recording", exist_ok=True)

        self.is_writing = True
        video_writer = cv2.VideoWriter("pgm/recording/video.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (800, 600))
        for frame in self.recording:
            video_writer.write(frame)
        video_writer.release()
        self.is_writing = False
    
    def get_progress(self):
        return self.lib.get_progress()
        # dpg.set_value(self.progress_bar, progress)
        # dpg.configure_item(self.progress_bar, overlay=f"{100 * progress:.2f}%")
        # dpg.set_value(self.eta_indicator, f"ETA: {eta:.2f}s")
    def get_eta(self):
        return self.lib.get_eta()

    def start_reconstruction(self):
        if self.running:
            return
        
        self.running = True

        os.makedirs("pgm", exist_ok=True)
        if os.path.exists("pgm/reconstruction"):
            shutil.rmtree("pgm/reconstruction")
        os.makedirs("pgm/reconstruction/model", exist_ok=True)

        self.is_reading = True

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
        
    def stop_reconstruction(self):
        if not self.running:
            return
        self.running = False
        self.lib.stop_photogrammetry_session()


    def __init__(self):
        self.is_recording = False
        self.is_writing = False
        self.is_reading = False
        self.recording = []
        self.running = False
        self.frame_rate = 30
          

        # label="Open in Preview", callback=lambda: os.system("open pgm/reconstruction/out.usdz")
        # label="Open in Meshlab", callback=lambda: os.system("open -a /Applications/MeshLab2023.12.app pgm/reconstruction/model/out.obj")

    def on_new_frame_callback(self, frame):
        if self.is_recording:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            self.recording.append(frame)

photogrammetry = Photogrammetry()
# camera(callback=on_new_frame_callback)