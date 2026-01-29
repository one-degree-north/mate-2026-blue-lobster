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
# Main Application
# =========================
def main():
    # ---- GLFW / OpenGL ----
    glfw.init()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

    window = glfw.create_window(1440, 900, "Dual Video View", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # ---- ImGui ----
    imgui.create_context()
    impl = GlfwRenderer(window)

    # ---- Video Stream ----
    recognizer = ImageRecognizer("assets/monkeydo.png")

    pipeline_desc = """
    udpsrc port=5000 caps="application/x-rtp, media=video, encoding-name=H264, payload=96" !
    rtph264depay ! avdec_h264 !
    videoconvert ! video/x-raw,format=RGB !
    appsink name=sink emit-signals=true max-buffers=1 drop=true
    """

    stream = VideoStream(pipeline_desc, recognizer)
    previous_time = time.time()
    start = False
    elapsed_time = 0

    # ---- Render Loop ----
    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()

        stream.update()
        tex_raw, tex_proc = stream.get_textures()
        shape = stream.get_frame_shape()

        io = imgui.get_io()
        imgui.set_next_window_size(io.display_size.x, io.display_size.y)

        imgui.begin(
            "Main",
            flags=(
                imgui.WINDOW_NO_TITLE_BAR
                | imgui.WINDOW_NO_RESIZE
                | imgui.WINDOW_NO_MOVE
                | imgui.WINDOW_NO_SCROLLBAR
            ),
        )
        imgui.begin_child("Videos", 400, 600)
        if shape:
            h, w = shape
            panel_w = 600
            panel_h = int(panel_w * h / w)

            imgui.begin_child("left_panel", panel_w, panel_h + 40, border=True)
            imgui.text("Raw Feed")
            imgui.image(tex_raw, panel_w, panel_h)
            imgui.end_child()


            # ---- Right panel ----
            imgui.begin_child("right_panel", panel_w, panel_h + 40, border=True)
            imgui.text("CV Detection")
            imgui.image(tex_proc, panel_w, panel_h)
            imgui.end_child()

        imgui.end_child()
        imgui.same_line()
        imgui.begin_child("Options", 1000, 1000, border=True)
        imgui.text("Options")

        imgui.begin_child("StopWatch", 500, 200, border=True)

        # Stop Watch
        imgui.text("Stop Watch")
        t = time.time()
        if imgui.button("Start"):
            start = not start
        
        if start:
            elapsed_time+=t-previous_time
        previous_time = t
        imgui.text(f"{int(elapsed_time//60):02d}:{int((elapsed_time%60)//1):02d}:{int(((elapsed_time%1)*1000)//1):03d}")        

        imgui.end_child()

        # 
        imgui.begin_child("Button 2", 500, 200)
        imgui.text("Button")
        imgui.end_child()

        imgui.end_child()
        imgui.end()

        imgui.render()
        gl.glClearColor(0.1, 0.1, 0.1, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    # ---- Shutdown ----
    stream.shutdown()
    impl.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    main()
