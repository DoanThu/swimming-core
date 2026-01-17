# How to run
## Find device's port
Run `python ./utils/find_camera.py` to find device's port.

Go to `config/general.py` to change `DEVICE_PORT` accordingly. `DEVICE_PORT` is 0 if use laptop camera, 1 if drone.

Change `DEVICE_PORT` to a video path to run a video.

## Download models
Download pose model [here](https://drive.google.com/drive/folders/1nhyLbYXetj_6qsy7T_fU9KqUF5aadUXI?usp=drive_link).

Download segmentation model [here](https://drive.google.com/drive/folders/1udW09qW0RTWQafEX3PXyl6kdNArFbd2l?usp=drive_link).

Go to `config/pose_model.yaml` and `config/seg_model.yaml` to change the model paths.

## Run the server
`python main.py --mode SOCKET_MULTI`

## Run UI
`streamlit run ui/swim_dashboard.py`