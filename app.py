import time
import gi
import concur as c
import datetime
import imgui
import os
import subprocess
import threading
import cv2

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from helper.monkeyseecrab import MultiCrabTracker
from video_stream import VideoStream
from photogrammetry.client import PhotoGrammetry, PgmBindings

def init_stream():
    tracker = MultiCrabTracker(detection_interval=20)
    tracker.load_training_image(
        "/Users/aryanmahajan/Documents/PROJECTS/mate-2026-blue-lobster/helper/reference.png"
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
            imgui.text(f" | Crabs Counted: {stream.get_count()}")
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
            if state["is_reconstructing"]: imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
            if imgui.button("Start Photogrammetry") and not state["is_reconstructing"]:
                state["photogrammetry_active"] = True
                state["photogrammetry"].start_recording()
            if state["is_reconstructing"]: imgui.pop_style_var()
        else:
            if imgui.button("Begin Reconstruction"):
                state["photogrammetry_active"] = False
                state["is_reconstructing"] = True
                state["start_time"] = time.time()
                state["photogrammetry"].stop_recording()

                def run_pgm():
                    try:
                        input_dir = os.path.abspath("pgm/reconstruction")
                        if not os.path.exists(input_dir) or not os.listdir(input_dir):
                            state["pgm_error"] = "Input directory missing or empty"
                            state["is_reconstructing"] = False
                            return
                        
                        # Call without the callback argument
                        state["photogrammetry"].start_reconstruction()
                    except Exception as e:
                        state["pgm_error"] = str(e)
                        state["is_reconstructing"] = False

                threading.Thread(target=run_pgm, daemon=True).start()

        if state["is_reconstructing"]:
            progress = state['photogrammetry'].get_progress()
            imgui.progress_bar(progress, size=(-1.0, 0.0), overlay=f"{100*progress:.1f}%")
            
            # Watchdog for the "Could not open file" system error
            if progress == 0 and (time.time() - state["start_time"] > 5.0):
                state["pgm_error"] = "Engine Timeout: Could not open files."
                state["is_reconstructing"] = False
            
            if progress >= 1.0 or state["photogrammetry"].is_completed():
                state["is_reconstructing"] = False

        if os.path.exists(model_path):
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.6, 0.2, 1.0)
            if imgui.button("Open Model in Preview"):
                subprocess.run(["open", "-a", "Preview", model_path])
            imgui.pop_style_color()

        yield

def main():
    stream = init_stream()
    state = {}
    state["photogrammetry"] = PhotoGrammetry(PgmBindings("libpgm.dylib"))

    c.main(c.orr([
        c.window("Processed Video", processed_video_panel(stream)),
        c.window("Options", options_panel(state, stream)),
    ]))
    stream.shutdown()

if __name__ == "__main__":
    main()