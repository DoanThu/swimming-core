# How to run (on the 4090 laptop)
## Environment
- Open terminal powershell
- Activate conda env `conda activate python`.

## Download models
Pose and segmentation models are stored in the SSD, folder `swimming_models`.
Go to `config/pose_model.yaml` and `config/seg_model.yaml` to change the model paths.

## Run video mode (for debugging)
In `config/general.py`:
- `VIDEO_PATH`: video to analyse
- `SAVE_CSV_PATH`: path to save csv file
- `SAVE_VIDEO_PATH`: path to save annotated video

Finally, run:
`python main.py --mode VIDEO_SINGLE` or `python main.py --mode VIDEO_MULTI`. Definition of each mode is written in file `main.py`.

## Run socket mode (for realtime demo with UI)
### Server
Turn on the core server by running:

`python main.py --mode SOCKET_MULTI`


In `config/general.py`:

- `DEVICE_ID` is usually 0 if using laptop webcam, 1 if using drone.
- Can change `DEVICE_ID` to `VIDEO_PATH` or any video path to debug the UI.
- Can change `RESOLUTION` or `FPS_RATE` to speed up.


When log shows **SERVER STARTED**, it is waiting for UI.
Open a new powershell tab:
- `conda activate python`
- `streamlit run ui/swim_dashboard.py` 

### Client
To see UI on any mobile device:
1. Make laptop as hotspot
2. Mobile deviced connected to that network
3. Go to **cmd**, run `ipconfig` to find the laptop IP
4. Open browser to access the UI @ *laptop_ip:port*, port is set in `SERVER_PORT` in `config/general.py`.

## Find device's port
Run `python ./utils/find_camera.py` to find device's port if it is unknown.
