import time
import gi
import concur as c
import datetime
import imgui
import os
import subprocess
import multiprocessing # Changed from threading
import cv2

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from helper.monkeyseecrab import MultiCrabTracker
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
    tracker.load_training_image("/Users/aryanmahajan/Documents/PROJECTS/mate-2026-blue-lobster/helper/reference.png")
    
    # Optimization: Added 'threads=2' to x264enc to leave CPU headroom for UI
    pipeline_desc = """
    udpsrc port=5000 caps="application/x-rtp, media=video, encoding-name=H264, payload=96" !
    rtph264depay ! avdec_h264 !
    videoconvert ! video/x-raw,format=RGB ! tee name=t
    t. ! queue ! appsink name=sink emit-signals=true max-buffers=1 drop=true
    """
    return VideoStream(pipeline_desc, tracker)

def options_panel(state, stream):
    if "is_reconstructing" not in state: state["is_reconstructing"] = False
    if "pgm_proc" not in state: state["pgm_proc"] = None
    
    model_path = os.path.abspath("pgm/reconstruction/out.usdz")

    while True:
        imgui.text("Options")
        imgui.separator()
        
        # Snapshot Button
        if imgui.button("Take Snapshot"):
            if stream.raw_frame is not None:
                os.makedirs("recordings", exist_ok=True)
                ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                cv2.imwrite(f"recordings/snapshot_{ts}.png", stream.raw_frame)
        
        imgui.separator()

        # Photogrammetry State Machine
        if state.get("photogrammetry_active"):
            if stream.raw_frame is not None:
                state["photogrammetry"].receive_frame(stream.raw_frame)

        if not state.get("photogrammetry_active"):
            # Disable button if process is running
            if state["is_reconstructing"]:
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
            
            if imgui.button("Start Photogrammetry") and not state["is_reconstructing"]:
                state["photogrammetry_active"] = True
                state["photogrammetry"].start_recording()
                
            if state["is_reconstructing"]: imgui.pop_style_var()
        else:
            if imgui.button("Begin Reconstruction"):
                state["photogrammetry_active"] = False
                state["is_reconstructing"] = True
                state["photogrammetry"].stop_recording()

                # Start the background process
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
            if imgui.button("Open Model in Preview"):
                subprocess.run(["open", "-a", "Preview", model_path])

        yield

def main():
    # Multiprocessing on macOS requires this inside the main guard
    multiprocessing.set_start_method('spawn', force=True)
    
    stream = init_stream()
    state = {}
    
    # The 'UI version' of photogrammetry for frame recording
    bindings = PgmBindings("libpgm.dylib")
    state["photogrammetry"] = PhotoGrammetry(bindings)

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