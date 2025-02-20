# How to run
## Find device's port
Run `python ./utils/find_camera.py` to find device's port.

Go to `config/general.py` to change `DEVICE_PORT` accordingly.

## Run the server
`python main.py --mode SOCKET`
<!-- 
# Where to start?
- Start from `main.py`
- General config is in `config/general.py`
- Input video and initial position of the swimmer is temporarily in `config/pose_model.yaml`
- The stats of each frame will be stored as json files -->
<!-- 
# Data
- Please upload your trained models [here](https://drive.google.com/drive/folders/19uo_DWLWtjJ31pi67yNlskhdLTiWDjj4).

- For testing, please use the videos [here](https://drive.google.com/drive/folders/1X-_0Hl811meSo-gxIBtga_3LPKEN9qKU). The nature of these videos is close to the real footage that will be sent to our system. One frame might have more than one swimmer. And the lane markers might be in any colors.

- If you do not have GPU, you can use a json file of keypoints returned from the YOLOv8 model [here](https://drive.google.com/drive/folders/1hveFL5civAPA96WANsPzauscKFyMqmH1?usp=drive_link). However, it is recommended to run this code with GPU. -->