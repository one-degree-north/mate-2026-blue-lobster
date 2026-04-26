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


# Instructions
Processed Video (Left \- Right):

- STATUS Label \- Whether the video feed is active or inactive (Buggy)  
- In Frame \- Number of crabs in frame  
- Total crabs \- Number of crabs seen since reset (Possibly buggy due to detection flickers)  
- Output FPS \- Frames per Second from Video Feed  
- Actual Video \- Video feed from the bot (Bounding boxes will appear for crab detection when enabled)

Options (Top \- Bottom):

- Crab Counter \- Section for Crab Counting Task  
  - Start Counter \- Starts the program for counting crabs (not online all the time due to performance)  
  - Stop Counter \- Stops the program for counting crabs (Use once done with counting)  
  - Reset Counter \- Resets the Total crabs to zero  
      
- Info Label \- Displays whether the counting program is active or not

- Recording Menu \- Section to record video segments and take screenshots  
  - Start/Stop Recording \- Toggles whether the video feed is being recorded  
    - The label next to it displays “Standby” when not recording, a timer for recording time, and a blinking red dot to indicate that the video feed is being recorded.  
  - Take Screenshot \- Takes a screenshot  
  - PS: Both are saved in the recordings file under the main directory

- Photogrammetry \- Section for Coral Modelling task  
  - Photogrammetry Capture Rate  
    - Photos per Second \- Slider for the photos per second being inputted to the photogrammetry algorithm (high values will make everything lag a lot) (Default \- 2\)  
    - The label beneath displays the current Photos per second for photogrammetry

  - Start Photogrammetry \- Toggles the photo photogrammetry program to process the video feed  
    - A progress bar will appear to show progress for photogrammetry  
  - Open Model in Preview \- Opens the last .ustz 3D Model from photogrammetry (Doesn’t do anything if photogrammetry hasn’t been done at least once)
