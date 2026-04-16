# Robotics

## Dependencies

Install everything inside `pyproject.toml`:
```
uv venv
source .venv/bin/activate
uv sync
```

Install the following:
```
brew install gstreamer just
```

## Running
```
just run-dev # Starts test gstreamer pipeline

just run-debug
just run
```