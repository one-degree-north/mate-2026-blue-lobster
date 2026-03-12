import ctypes
import cv2
import os
import shutil
import time
from typing import Callable

class PgmBindings:
    def __init__(self, path: str) -> None:
        self.lib = ctypes.cdll.LoadLibrary(path)

        self.lib.run_photogrammetry_session.argtypes = [ctypes.c_char_p]
        self.lib.run_photogrammetry_session.restype = None

        self.lib.is_completed.argtypes = []
        self.lib.is_completed.restype = ctypes.c_bool

        self.lib.get_progress.argtypes = []
        self.lib.get_progress.restype = ctypes.c_double

        self.lib.get_eta.argtypes = []
        self.lib.get_eta.restype = ctypes.c_double

        self.lib.stop_photogrammetry_session.argtypes = []
        self.lib.stop_photogrammetry_session.restype = None
    
    def start(self, path: bytes) -> None:
        return self.lib.run_photogrammetry_session(path)
    
    def stop(self) -> None:
        return self.lib.stop_photogrammetry_session()
    
    def is_completed(self) -> bool:
        return self.lib.is_completed()
    
    def get_progress(self) -> float:
        return self.lib.get_progress()
    
    def get_eta(self) -> float:
        return self.lib.get_eta()
    


class Photogrammetry:
    def __init__(self, lib: PgmBindings) -> None:
        self.lib = lib
        
        self.recording: list[cv2.typing.MatLike] = []
        self.frame_rate: int = 30

    
    def receive_frame(self, frame: cv2.typing.MatLike) -> None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        self.recording.append(frame)

    def start_recording(self) -> None:
        self.recording = []

    def stop_recording(self):
        os.makedirs("pgm", exist_ok=True)

        if os.path.exists("pgm/recording"):
            shutil.rmtree("pgm/recording")

        os.makedirs("pgm/recording", exist_ok=True)

        if not self.recording:
            return

        frame_height, frame_width = self.recording[0].shape[:2]
        video_writer = cv2.VideoWriter("pgm/recording/video.avi", cv2.VideoWriter_fourcc(*"MJPG"), self.frame_rate, (frame_width, frame_height))
        for frame in self.recording:
            video_writer.write(frame)
        video_writer.release()
    
    def start_reconstruction(self):
        os.makedirs("pgm", exist_ok=True)
        if os.path.exists("pgm/reconstruction"):
            shutil.rmtree("pgm/reconstruction")
        os.makedirs("pgm/reconstruction/model", exist_ok=True)


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
            
        # frames = len(os.listdir("pgm/recording"))
        # skip_frames = max(1, 30 // self.frame_rate)
        # for frame in range(1, frames + 1, skip_frames):
        #     shutil.copy(f"pgm/recording/frame_{frame}.png", f"pgm/reconstruction/frame_{frame}.png")

        self.lib.start(b"pgm/reconstruction")
        
    def stop_reconstruction(self):
        self.lib.stop()
    
    def set_frame_rate(self, _, frame_rate):
        self.frame_rate = frame_rate

    def get_progress(self):
        return self.lib.get_progress()
        # dpg.set_value(self.progress_bar, progress)
        # dpg.configure_item(self.progress_bar, overlay=f"{100 * progress:.2f}%")
        # dpg.set_value(self.eta_indicator, f"ETA: {eta:.2f}s")

    def get_eta(self):
        return self.lib.get_eta()

    def is_completed(self):
        return self.lib.is_completed()
    

if __name__ == "__main__":

    bindings = PgmBindings("libpgm.dylib")
    photogrammetry = Photogrammetry(bindings)
    """
    photogrammetry.start_recording()

    cap = cv2.VideoCapture("ignore/test.mov")
    while True:
        success, frame = cap.read()
        if not success:
            break
        photogrammetry.receive_frame(frame)
    
    photogrammetry.stop_recording()
    photogrammetry.start_reconstruction()
    """
    print("Reconstruction started...")

    while True:
        time.sleep(0.5)
        progress = photogrammetry.get_progress()
        eta = photogrammetry.get_eta()
        print(f"Progress: {progress*100:.1f}%  ETA: {eta:.1f}s", end="\r", flush=True)

        if photogrammetry.is_completed():
            break

    print("Reconstruction complete!")


    # camera(callback=on_new_frame_callback)

    # label="Open in Preview", callback=lambda: os.system("open pgm/reconstruction/out.usdz")
    # label="Open in Meshlab", callback=lambda: os.system("open -a /Applications/MeshLab2023.12.app pgm/reconstruction/model/out.obj")