import time
import gi
import concur as c
import datetime
import imgui
import os
import subprocess
import multiprocessing
import cv2

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from .monkeyseecrab import MultiCrabTracker
from video_stream import VideoStream
from photogrammetry.client import PhotoGrammetry, PgmBindings

# --- Background Worker Function ---
def run_reconstruction_worker():
    """
    Runs in a completely separate OS process to prevent UI lag.
    """
    # Lower the priority of this process so the UI stays smooth
    try:
        os.nice(10) 
    except:
        pass

    # We must re-init the library inside the new process
    bindings = PgmBindings("libpgm.dylib")
    pgm = PhotoGrammetry(bindings)
    
    # Check if files exist before starting
    input_dir = os.path.abspath("pgm/reconstruction")
    if os.path.exists(input_dir) and len(os.listdir(input_dir)) > 0:
        pgm.start_reconstruction()
        
        # Keep process alive while engine is working
        while not pgm.is_completed():
            time.sleep(0.5)

# --- UI Panels ---

def init_stream():
    tracker = MultiCrabTracker(detection_interval=20)
    tracker.load_training_image(
        os.path.join(os.path.dirname(__file__), "reference.png")
    )
    tracker.counting = False
    pipeline_desc = """
    udpsrc port=5000 caps="application/x-rtp, media=video, encoding-name=H264, payload=96" !
    rtph264depay ! avdec_h264 !
    videoconvert ! video/x-raw,format=RGB ! tee name=t
    t. ! queue ! appsink name=sink emit-signals=true max-buffers=1 drop=true
    """
    return VideoStream(pipeline_desc, tracker)

def set_global_font_once():
    io = imgui.get_io()
    io.font_global_scale = 1.5 

def snap_to_grid(size=20):
    x, y = imgui.get_window_position()
    imgui.set_window_position(round(x / size) * size, round(y / size) * size)

def processed_video_panel(stream):
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
            imgui.text_colored(f"STATUS: TRACKING", 0.3, 1.0, 0.3, 1.0)
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

def video_recording_panel(state, stream):
    if "recording" not in state: state["recording"] = False
    current_time = time.time()
    button_label = "STOP Recording" if state["recording"] else "START Recording"
    
    if imgui.button(button_label):
        if not state["recording"]:
            state["active_bin"], state["active_pad"] = stream.start_recording()
            state["rec_start_time"], state["recording"] = current_time, True
        else:
            stream.stop_recording(state.get("active_bin"), state.get("active_pad"))
            state["recording"] = False

    imgui.same_line()
    if state["recording"]:
        if int(current_time * 2) % 2 == 0: imgui.text_colored("REC", 1.0, 0.0, 0.0, 1.0)
        else: imgui.text("   ")
        imgui.same_line()
        elapsed = current_time - state["rec_start_time"]
        imgui.text(f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}")
    else:
        imgui.text_disabled("Standby")

def options_panel(state, stream):
    if "photogrammetry_active" not in state: state["photogrammetry_active"] = False
    if "is_reconstructing" not in state: state["is_reconstructing"] = False
    if "pgm_error" not in state: state["pgm_error"] = None
    if "start_time" not in state: state["start_time"] = 0
    if "counter_running" not in state: state["counter_running"] = False
    if "pgm_capture_hz" not in state: state["pgm_capture_hz"] = 2.0
    if "pgm_proc" not in state: state["pgm_proc"] = None
    
    model_path = os.path.abspath("pgm/reconstruction/out.usdz")

    while True:
        snap_to_grid(50)
        imgui.text("Options")
        imgui.separator()

        imgui.text("Crab Counter")
        if imgui.button("Start Counter"):
            state["counter_running"] = True
        imgui.same_line()
        if imgui.button("Stop Counter"):
            state["counter_running"] = False
        imgui.same_line()
        if imgui.button("Reset Counter"):
            stream.reset_counter()

        stream.set_detection_active(state["counter_running"])

        if state["counter_running"]:
            imgui.text_colored("Counter armed: hold button to count", 0.2, 0.9, 0.2, 1.0)
            imgui.button("HOLD TO COUNT")
            stream.set_counting_active(imgui.is_item_active())
        else:
            imgui.text_disabled("Counter stopped")
            stream.set_counting_active(False)
        
        imgui.separator()
        
        video_recording_panel(state, stream)
        imgui.spacing()

        if imgui.button("Take Snapshot"):
            if stream.raw_frame is not None:
                if not os.path.isdir("recordings"):
                    os.makedirs("recordings/")
                ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"recordings/snapshot_{ts}.png"
                cv2.imwrite(filename, stream.raw_frame)
        
        imgui.separator()

        imgui.text("Photogrammetry Capture Rate")
        _, state["pgm_capture_hz"] = imgui.slider_float(
            "Photos per second",
            state["pgm_capture_hz"],
            0.2,
            10.0,
            "%.1f"
        )
        state["photogrammetry"].set_capture_rate(state["pgm_capture_hz"])
        imgui.text_disabled(f"Current: {state['pgm_capture_hz']:.1f} photos/sec")
        imgui.separator()

        if state["pgm_error"]:
            imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 0.4, 0.4, 1.0)
            imgui.text_wrapped(f"ENGINE ERROR: {state['pgm_error']}")
            imgui.pop_style_color()
            if imgui.button("Dismiss Error"): state["pgm_error"] = None
            imgui.separator()

        if state["photogrammetry_active"]:
            if stream.raw_frame is not None:
                state["photogrammetry"].receive_frame(stream.raw_frame)

        if not state["photogrammetry_active"]:
            if state["is_reconstructing"] or state["pgm_proc"]: 
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
            if imgui.button("Start Photogrammetry") and not state["is_reconstructing"]:
                state["photogrammetry_active"] = True
                state["photogrammetry"].start_recording()
            if state["is_reconstructing"] or state["pgm_proc"]: 
                imgui.pop_style_var()
        else:
            if imgui.button("Begin Reconstruction"):
                state["photogrammetry_active"] = False
                state["is_reconstructing"] = True
                state["start_time"] = time.time()
                state["photogrammetry"].stop_recording()

                # Start the background process (multiprocessing for better UI performance)
                proc = multiprocessing.Process(target=run_reconstruction_worker)
                proc.start()
                state["pgm_proc"] = proc

        # Monitor the Background Process
        if state["is_reconstructing"] and state["pgm_proc"]:
            imgui.text_colored("Engine running in background...", 0.4, 0.7, 1.0, 1.0)
            
            # Check if process is still alive
            if not state["pgm_proc"].is_alive():
                state["is_reconstructing"] = False
                state["pgm_proc"] = None
        
        if os.path.exists(model_path):
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.6, 0.2, 1.0)
            if imgui.button("Open Model in Preview"):
                subprocess.run(["open", "-a", "Preview", model_path])
            imgui.pop_style_color()

        yield

def main():
    # Multiprocessing on macOS requires this inside the main guard
    multiprocessing.set_start_method('spawn', force=True)
    
    stream = init_stream()
    state = {}
    state["photogrammetry"] = PhotoGrammetry(PgmBindings("libpgm.dylib"))

    c.main(c.orr([
        c.window("Processed Video", processed_video_panel(stream)),
        c.window("Options", options_panel(state, stream)),
    ]))
    
    # Cleanup
    if state.get("pgm_proc"):
        state["pgm_proc"].terminate()
    stream.shutdown()

if __name__ == "__main__":
    main()