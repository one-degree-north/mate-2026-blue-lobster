import time
import gi

import concur as c
import datetime
import imgui
import os

from helper.monkeyseecrab import ImageRecognizer
from video_stream import VideoStream

gi.require_version("Gst", "1.0")
from gi.repository import Gst
from photogrammetry import PhotoGrammetry

# =========================
# Video stream
# =========================

def init_stream():
    recognizer = ImageRecognizer("assets/monkeydo.png")
    pipeline_desc = """
    udpsrc port=5000 caps="application/x-rtp, media=video, encoding-name=H264, payload=96" !
    rtph264depay ! avdec_h264 !
    videoconvert ! video/x-raw,format=RGB ! tee name=t
    t. ! queue ! appsink name=sink emit-signals=true max-buffers=1 drop=true
    """
    return VideoStream(pipeline_desc, recognizer)

# =========================
# Init font
# =========================

def set_global_font_once():
    io = imgui.get_io()
    io.font_global_scale = 1.5  # 1.2, 1.5, 2.0



# =========================
# UI Widgets
# =========================

def snap_to_grid(size=20):
    x, y = imgui.get_window_position()
    snapped_x = round(x / size) * size
    snapped_y = round(y / size) * size
    imgui.set_window_position(snapped_x, snapped_y)


def raw_video_panel(stream):
    set_global_font_once()
    while True:
        snap_to_grid(50)

        stream.update()
        tex_raw, _, _ = stream.get_textures()
        shape = stream.get_frame_shape()

        if shape:
            h, w = shape
            avail_w, avail_h = imgui.get_content_region_available()
            ratio = min(avail_w / w, avail_h / h)
            panel_w = int(w * ratio)
            panel_h = int(h * ratio)

            imgui.image(tex_raw, panel_w, panel_h)

        yield


def processed_video_panel(stream):
    while True:
        snap_to_grid(50)

        stream.update()
        _, tex_proc, _ = stream.get_textures()
        shape = stream.get_frame_shape()

        if shape:
            h, w = shape
            avail_w, avail_h = imgui.get_content_region_available()
            ratio = min(avail_w / w, avail_h / h)
            panel_w = int(w * ratio)
            panel_h = int(h * ratio)

            imgui.image(tex_proc, panel_w, panel_h)

        yield

def third_video_panel(stream):
    while True:
        snap_to_grid(50)

        stream.update()
        _, _, tex_third = stream.get_textures()
        shape = stream.get_frame_shape()

        if shape:
            h, w = shape
            avail_w, avail_h = imgui.get_content_region_available()
            ratio = min(avail_w / w, avail_h / h)
            panel_w = int(w * ratio)
            panel_h = int(h * ratio)

            imgui.image(tex_third, panel_w, panel_h)

        yield


def video_recording_panel(state, stream):
    # --- Initialization ---
    if "recording" not in state:
        state["recording"] = False
    
    # --- Cooldown Logic ---
    current_time = time.time()

    button_label = "STOP Recording" if state["recording"] else "START Recording"
    
    if imgui.button(button_label):
        
        if not state["recording"]:
            # START
            state["active_bin"], state["active_pad"] = stream.start_recording()
            state["rec_start_time"] = current_time
            state["recording"] = True
        else:
            # STOP
            # Ensure we call the cleanup with EOS in your VideoStream class
            stream.stop_recording(state.get("active_bin"), state.get("active_pad"))
            state["recording"] = False
            state["active_bin"] = None
            state["active_pad"] = None

    imgui.same_line()

    # --- Indicators ---
    if state["recording"]:
        # Blinking REC icon
        if int(current_time * 2) % 2 == 0:
            imgui.text_colored("REC", 1.0, 0.0, 0.0, 1.0)
        else:
            imgui.text("   ")
        
        imgui.same_line()
        
        elapsed = current_time - state["rec_start_time"]
        imgui.text(f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}")
    else:
        imgui.text_disabled("Standby")

def options_panel(state, stream):
    while True:
        snap_to_grid(50)
        imgui.text("Options")
        imgui.separator()
        imgui.spacing()

        video_recording_panel(state, stream)
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        if imgui.button("Take Snapshot"):
            if stream.raw_frame is not None:
                if not os.path.isdir("recordings"):
                    os.makedirs("recordings/")
                import cv2
                ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"recordings/snapshot_{ts}.png"
                cv2.imwrite(filename, stream.raw_frame)
                print(f"Snapshot saved: {filename}")
        
        imgui.spacing()
        
        imgui.separator()
        imgui.spacing()

        if "photogrammetry_active" not in state or not state["photogrammetry_active"]:
            state["photogrammetry_active"] = False
        if not state["photogrammetry_active"]:
            if imgui.button("Start Photogrammetary"):
                state["photogrammetry_active"] = True
                print("Starting photogrammetry...")

                # Optionally, capture a few frames for reconstruction
                import threading

                def run_photogrammetry():
                    # Example: send the latest raw frame
                    if stream.raw_frame is not None:
                        state["photogrammetry"].send_iframe(stream.raw_frame)

                threading.Thread(target=run_photogrammetry, daemon=True).start()
        else:
            if imgui.button("Begin Reconstruction"):
                print("ending photogrammetry...")
                state["photogrammetry_active"] = False

                # Capture frames asynchronously
                def end_callback():
                    state["photogrammetry_active"] = False
                    print("Photogrammetry reconstruction complete!")

                # Optionally, capture a few frames for reconstruction
                import threading

                def start_reconstruction():
                    # Start the (mock) reconstruction
                    state["photogrammetry"].start_reconstruction(end_callback)

                threading.Thread(target=start_reconstruction, daemon=True).start()

        yield
        

# =========================
# Main
# =========================

def main():
    stream = init_stream()
    state = {}
    state["photogrammetry"] = PhotoGrammetry()

    c.main(
        c.orr([
            c.window("Raw Video", raw_video_panel(stream)),
            c.window("Processed Video", processed_video_panel(stream)),
            c.window("Third Video", third_video_panel(stream)),
            c.window("Options", options_panel(state, stream)),
        ])
    )

    stream.shutdown()


if __name__ == "__main__":
    main()