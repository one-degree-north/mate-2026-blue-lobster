# Justfile for mate-26 project

set shell := ["bash", "-uc"]

DYLD := "/opt/homebrew/opt/gstreamer/lib:/opt/homebrew/opt/glib/lib:/opt/homebrew/opt/gobject-introspection/lib"

# Default recipe
default: run

# Run the application
run:
    #!/bin/bash
    export DYLD_LIBRARY_PATH="{{DYLD}}"
    python -m app.app

# Run with debug output
run-debug:
    #!/bin/bash
    export DYLD_LIBRARY_PATH="{{DYLD}}"
    python -m app.app --debug

# Stream video from camera in background, then run app in debug mode
run-dev:
    #!/bin/bash
    export DYLD_LIBRARY_PATH="{{DYLD}}"
    gst-launch-1.0 avfvideosrc! videoconvert ! video/x-raw,format=I420 ! x264enc tune=zerolatency speed-preset=ultrafast bitrate=5000 key-int-max=30 ! rtph264pay pt=96 config-interval=1 ! udpsink host=127.0.0.1 port=5000 &
    GST_PID=$!
    trap 'kill "$GST_PID" 2>/dev/null || true' EXIT
    python -m app.app --debug
