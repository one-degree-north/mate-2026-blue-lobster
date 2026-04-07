import ctypes
import os
import shutil
import sys
import time
from typing import Callable

import cv2


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
    def __init__(
        self,
        lib: PgmBindings,
        input_fps = 15,
        output_fps: int = 30,
        output_size: tuple[int, int] = "auto", # (width, height)
    ) -> None:
        self.lib = lib
        self.frame_rate: int = input_fps
        self.target_fps = output_fps
            
        self.target_size = output_size # detect auto later
        self._frame_count = 0
        self._saved_frame_count = 0
        self._is_recording = False

    def _reset_dir(self, path: str) -> None:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    def receive_frame(self, frame: cv2.typing.MatLike) -> None:

        if not self._is_recording:
            return

        if self.target_size == "auto":
            self.target_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        skip_interval = max(1, round(self.frame_rate / self.target_fps))

        if self._frame_count % skip_interval == 0:
            #frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            frame = cv2.resize(frame, self.target_size)
            self._saved_frame_count += 1
            cv2.imwrite(
                f"pgm/reconstruction/frame_{self._saved_frame_count}.png",
                frame,
            )

        self._frame_count += 1

    def start_recording(self) -> None:
        self._reset_dir("pgm/reconstruction")
        self._frame_count = 0
        self._saved_frame_count = 0
        self._is_recording = True

    def stop_recording(self):
        self._is_recording = False
        if self._saved_frame_count == 0:
            print("Warning: no frames were recorded.")

    def start_reconstruction(self):
        os.makedirs("pgm/reconstruction/model", exist_ok=True)
        self.lib.start(b"pgm/reconstruction")

    def stop_reconstruction(self):
        self.lib.stop()

    def set_frame_rate(self, _, frame_rate):
        self.frame_rate = frame_rate

    def get_progress(self):
        return self.lib.get_progress()

    def get_eta(self):
        return self.lib.get_eta()

    def is_completed(self):
        return self.lib.is_completed()


if __name__ == "__main__":
    bindings = PgmBindings("libpgm.dylib")
    photogrammetry = Photogrammetry(bindings, input_fps=30, output_fps=30)

    if "--skip-cv" not in sys.argv:
        photogrammetry.start_recording()

        cap = cv2.VideoCapture(sys.argv[1])

        while True:
            success, frame = cap.read()
            if not success:
                break
            photogrammetry.receive_frame(frame)

        cap.release()
        photogrammetry.stop_recording()

    photogrammetry.start_reconstruction()

    # TODO: Use tqmz. and make API more pythononic -- then also expose an feed driven api

    print("Reconstruction started...")

    while True:
        time.sleep(0.1)
        progress = photogrammetry.get_progress()
        eta = photogrammetry.get_eta()
        print(f"Progress: {progress * 100:.1f}%  ETA: {eta:.1f}s", end="\r", flush=True)

        if photogrammetry.is_completed():
            break

    print("Reconstruction complete!")
