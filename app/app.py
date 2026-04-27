import datetime
import os
import subprocess
import threading
import time
from collections.abc import Generator
from dataclasses import dataclass

import concur as c
import cv2
import gi
import imgui

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from pgm import Photogrammetry, BoundingBox

from .monkeyseecrab import MultiCrabTracker
from .video_stream import VideoStream

TEMP_DIR = "pgm-temp"
OUTPUT_PATH = "output.usdz"
SCALE_FACTOR = 1.0


@dataclass
class AppState:
    photogrammetry: Photogrammetry
    stop_event: threading.Event

    progress: float = 0.0

    recording: bool = False
    active_bin: Gst.Bin | None = None
    active_pad: Gst.Pad | None = None
    rec_start_time: float | None = None

    photogrammetry_active: bool = False
    is_reconstructing: bool = False
    pgm_error: str | None = None
    start_time: float = 0.0
    counter_running: bool = False
    pgm_capture_hz: float = 2.0
    pgm_thread: threading.Thread | None = None

    scale_factor: float = SCALE_FACTOR
    bounding_box: BoundingBox | None = None


# --- Background Worker Function ---
def run_reconstruction_worker(photogrammetry: Photogrammetry, app_state: AppState, stop_event: threading.Event) -> None:
    """
    Runs in a background thread to prevent UI lag.
    """
    # Lower the priority of this thread so the UI stays smooth
    try:
        os.nice(10)
    except:
        pass

    photogrammetry.start_reconstruction()

    # Keep thread alive while engine is working
    while not photogrammetry.is_completed() and not stop_event.is_set():
        app_state.progress = photogrammetry.get_progress()
        time.sleep(0.5)


# --- UI Panels ---


def init_stream() -> VideoStream:
    tracker = MultiCrabTracker()
    tracker.load_training_image(os.path.join(os.path.dirname(__file__), "..", "assets", "monkeydo.png"))
    tracker.counting = False
    pipeline_desc = """
    udpsrc port=5000 caps="application/x-rtp, media=video, encoding-name=H264, payload=96" !
    rtph264depay ! avdec_h264 !
    videoconvert ! video/x-raw,format=RGB ! tee name=t
    t. ! queue ! appsink name=sink emit-signals=true max-buffers=1 drop=true
    """
    return VideoStream(pipeline_desc, tracker)


def set_global_font_once() -> None:
    io = imgui.get_io()
    io.font_global_scale = 1.5


def snap_to_grid(size: int = 20) -> None:
    x, y = imgui.get_window_position()
    imgui.set_window_position(round(x / size) * size, round(y / size) * size)


def processed_video_panel(stream: VideoStream) -> Generator[None, None, None]:
    set_global_font_once()
    while True:
        snap_to_grid(50)
        stream.update()
        textures = stream.get_textures()
        tex_proc = textures[1] if textures else None
        shape = stream.get_frame_shape()

        if shape and tex_proc is not None:
            h, w = shape
            avail_w, avail_h = imgui.get_content_region_available()
            imgui.text_colored("STATUS: TRACKING", 0.3, 1.0, 0.3, 1.0)
            imgui.same_line()
            imgui.text(f" | In Frame: {stream.get_visible_count()}")
            imgui.same_line()
            imgui.text(f" | Total Crabs: {stream.get_count()}")
            imgui.same_line()
            imgui.text(f" | Output FPS: {stream.get_output_fps():.1f}")
            ratio = min(avail_w / w, (avail_h - 30) / h)
            imgui.image(tex_proc, int(w * ratio), int(h * ratio))
        else:
            imgui.text_disabled("Searching for signal...")
        yield


def video_recording_panel(state: AppState, stream: VideoStream) -> None:
    current_time = time.time()
    button_label = "STOP Recording" if state.recording else "START Recording"

    if imgui.button(button_label):
        if not state.recording:
            state.active_bin, state.active_pad = stream.start_recording()
            state.rec_start_time, state.recording = current_time, True
        else:
            stream.stop_recording(state.active_bin, state.active_pad)
            state.recording = False

    imgui.same_line()
    if state.recording:
        if int(current_time * 2) % 2 == 0:
            imgui.text_colored("REC", 1.0, 0.0, 0.0, 1.0)
        else:
            imgui.text("   ")
        imgui.same_line()
        elapsed = current_time - state.rec_start_time
        imgui.text(f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}")
    else:
        imgui.text_disabled("Standby")


def options_panel(state: AppState, stream: VideoStream) -> Generator[None, None, None]:
    model_path = os.path.abspath(OUTPUT_PATH)

    while True:
        snap_to_grid(50)
        imgui.text("Options")
        imgui.separator()

        imgui.text("Crab Counter")
        if imgui.button("Start Counter"):
            state.counter_running = True
        imgui.same_line()
        if imgui.button("Stop Counter"):
            state.counter_running = False
        imgui.same_line()
        if imgui.button("Reset Counter"):
            stream.reset_counter()

        stream.set_detection_active(state.counter_running)

        stream.set_counting_active(state.counter_running)
        if state.counter_running:
            imgui.text_colored("Counting...", 0.2, 0.9, 0.2, 1.0)
        else:
            imgui.text_disabled("Counter stopped")

        imgui.separator()

        video_recording_panel(state, stream)
        imgui.spacing()

        if imgui.button("Take Snapshot"):
            if stream.raw_frame is not None:
                if not os.path.isdir("recordings"):
                    os.makedirs("recordings/")
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recordings/snapshot_{ts}.png"
                cv2.imwrite(filename, cv2.cvtColor(stream.raw_frame, cv2.COLOR_RGB2BGR))

        imgui.separator()

        imgui.text("Photogrammetry Capture Rate")
        if state.photogrammetry_active:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
        changed, state.pgm_capture_hz = imgui.slider_float("Photos per second", state.pgm_capture_hz, 0.2, 10.0, "%.1f")
        if state.photogrammetry_active:
            imgui.pop_style_var()
            imgui.text_disabled("(locked during recording)")
        elif changed:
            state.photogrammetry.set_capture_rate(state.pgm_capture_hz)
        imgui.text_disabled(f"Current: {state.pgm_capture_hz:.1f} photos/sec")
        imgui.separator()

        imgui.text("Photogrammetry Video FPS")
        if state.photogrammetry_active:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
        _, state.photogrammetry.video_fps = imgui.slider_int("Video FPS", state.photogrammetry.video_fps, 1, 60)
        if state.photogrammetry_active:
            imgui.pop_style_var()
            imgui.text_disabled("(locked during recording)")
        imgui.separator()

        imgui.text("Photogrammetry Detail Level")
        if state.photogrammetry_active:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
        _, state.photogrammetry.detail = imgui.slider_int("Detail Level", state.photogrammetry.detail, 0, 4)
        if state.photogrammetry_active:
            imgui.pop_style_var()
            imgui.text_disabled("(locked during recording)")
        imgui.separator()

        if state.pgm_error:
            imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 0.4, 0.4, 1.0)
            imgui.text_wrapped(f"ENGINE ERROR: {state.pgm_error}")
            imgui.pop_style_color()
            if imgui.button("Dismiss Error"):
                state.pgm_error = None
            imgui.separator()

        if state.photogrammetry_active:
            if stream.raw_frame is not None:
                state.photogrammetry.receive_frame(stream.raw_frame)

        if not state.photogrammetry_active:
            if state.is_reconstructing or state.pgm_thread:
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
            if imgui.button("Start Photogrammetry") and not state.is_reconstructing:
                state.photogrammetry_active = True
                state.photogrammetry.start_recording()
            if state.is_reconstructing or state.pgm_thread:
                imgui.pop_style_var()
        else:
            if imgui.button("Begin Reconstruction"):
                state.photogrammetry_active = False
                state.is_reconstructing = True
                state.start_time = time.time()
                state.photogrammetry.stop_recording()

                # Start the background thread for reconstruction
                state.stop_event.clear()
                thread = threading.Thread(
                    target=run_reconstruction_worker, args=(state.photogrammetry, state, state.stop_event), daemon=True
                )
                thread.start()
                state.pgm_thread = thread

        # Monitor the Background Thread
        if state.is_reconstructing and state.pgm_thread:
            imgui.text_colored("Engine running in background...", 0.4, 0.7, 1.0, 1.0)
            imgui.progress_bar(state.progress, size=(-1, 20))

            # Check if thread is still alive
            if not state.pgm_thread.is_alive():
                state.is_reconstructing = False
                state.pgm_thread = None

        if os.path.exists(model_path):
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.6, 0.2, 1.0)
            if imgui.button("Open Model in Preview"):
                subprocess.run(["open", "-a", "Preview", model_path])
            imgui.pop_style_color()

        imgui.separator()
        imgui.text("Bounding Box")
        _, state.scale_factor = imgui.input_float("Scale Factor", state.scale_factor, step=0.1, format="%.3f")

        can_get_bbox = os.path.exists(model_path) or state.photogrammetry.is_completed()
        if not can_get_bbox:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
        if imgui.button("Get Bounding Box") and can_get_bbox:
            state.bounding_box = state.photogrammetry.get_bounding_box(state.scale_factor)
        if not can_get_bbox:
            imgui.pop_style_var()

        if state.bounding_box is not None:
            bb = state.bounding_box
            imgui.text(f"W: {bb.width:.4f}  H: {bb.height:.4f}  D: {bb.depth:.4f}")
            imgui.text_disabled(f"({bb.min_x:.3f}, {bb.min_y:.3f}, {bb.min_z:.3f}) → ({bb.max_x:.3f}, {bb.max_y:.3f}, {bb.max_z:.3f})")

        yield


def main() -> None:
    stream = init_stream()
    state = AppState(
        photogrammetry=Photogrammetry(
            video_fps=30, target_fps=15, detail=0, temp_dir=TEMP_DIR, output_path=OUTPUT_PATH
        ),
        stop_event=threading.Event(),
    )

    c.main(
        c.orr(
            [
                c.window("Processed Video", processed_video_panel(stream)),
                c.window("Options", options_panel(state, stream)),
            ]
        )
    )

    # Cleanup
    if state.pgm_thread and state.pgm_thread.is_alive():
        state.stop_event.set()
        state.pgm_thread.join(timeout=2.0)
    stream.shutdown()


if __name__ == "__main__":
    main()
