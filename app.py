import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer
import OpenGL.GL as gl
import gi
import time

from helper.monkeyseecrab import ImageRecognizer
from video_stream import VideoStream

gi.require_version("Gst", "1.0")


# =========================
# Initialization
# =========================

def init_window():
    glfw.init()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

    window = glfw.create_window(1440, 900, "Dual Video View", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    return window


def init_imgui(window):
    imgui.create_context()
    return GlfwRenderer(window)


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
# UI Rendering
# =========================

def render_video_section(stream):
    stream.update()
    tex_raw, tex_proc = stream.get_textures()
    shape = stream.get_frame_shape()

    imgui.begin_child("Videos", 400, 800)

    if shape:
        h, w = shape
        panel_w = 400
        panel_h = int(panel_w * h / w)

        # Left panel
        imgui.begin_child("left_panel", panel_w, panel_h + 40, border=True)
        imgui.text("Raw Feed")
        imgui.image(tex_raw, panel_w, panel_h)
        imgui.end_child()


        # Right panel
        imgui.begin_child("right_panel", panel_w, panel_h + 40, border=True)
        imgui.text("CV Detection")
        imgui.image(tex_proc, panel_w, panel_h)
        imgui.end_child()

    imgui.end_child()


def render_stopwatch(state):
    imgui.begin_child("StopWatch", 500, 200, border=True)

    imgui.text("Stop Watch")

    t = time.time()

    if imgui.button("Start"):
        state["running"] = not state["running"]

    if state["running"]:
        state["elapsed"] += t - state["prev_time"]

    state["prev_time"] = t

    elapsed = state["elapsed"]
    imgui.text(
        f"{int(elapsed//60):02d}:"
        f"{int((elapsed%60)//1):02d}:"
        f"{int(((elapsed%1)*1000)//1):03d}"
    )

    imgui.end_child()


def render_options_section(state):
    imgui.begin_child("Options", 0, 0, border=True)
    imgui.text("Options")

    render_stopwatch(state)

    imgui.begin_child("Button 2", 500, 200)
    imgui.text("Button")
    imgui.end_child()

    imgui.end_child()


# =========================
# Main Loop
# =========================

def main_loop(window, impl, stream):
    state = {
        "running": False,
        "elapsed": 0,
        "prev_time": time.time(),
    }

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()

        io = imgui.get_io()
        imgui.set_next_window_size(io.display_size.x-100, io.display_size.y-100)

        imgui.begin(
            "Main",
            flags=(
                imgui.WINDOW_NO_TITLE_BAR
                | imgui.WINDOW_NO_RESIZE
                | imgui.WINDOW_NO_MOVE
                | imgui.WINDOW_NO_SCROLLBAR
            ),
        )

        render_video_section(stream)

        imgui.same_line()

        render_options_section(state)

        imgui.end()

        imgui.render()
        gl.glClearColor(0.1, 0.1, 0.1, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)


# =========================
# Entry Point
# =========================

def main():
    window = init_window()
    impl = init_imgui(window)
    stream = init_stream()

    main_loop(window, impl, stream)

    stream.shutdown()
    impl.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    main()