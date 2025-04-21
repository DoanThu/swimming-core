SAVE_MODE = 'VIDEO' 
# SAVE_MODE = 'SOCKET'

FPS_RATE = 2 # read every n frames, needs to be a factor of with freq_frame_idx in seg_model.yaml
MAP_FPS_RATE_FREQ_SEGMENT = {2: 30, 3:30, 4:32}
FREQ_SEGMENT = MAP_FPS_RATE_FREQ_SEGMENT[FPS_RATE] # call the seg model every n frames
NO_POINTS_SEGMENTATION = 30


# Model
POSE_CONFIG = 'config/pose_model.yaml'
SEG_CONFIG = 'config/seg_model.yaml'
DETECT_CONFIG = 'config/detect_model.yaml'
FACE_CONFIG = 'config/face_model.yaml'
OPTICAL_FLOW_CONFIG = 'config/optical_flow_model.yaml'

# Sytem 
SAVE_AFTER_SECONDS = 5 
SAVE_JSON_PATH = 'frame_info/interval_{}.json'
SAVE_CSV_COLUMNS = ['timestamp', 'speed', 'speed_pct_change', 'pct_dist_changes', 'angle_changes']

# For VIDEO
FILENAME = 'DJI_0221.MP4'
VIDEO_PATH = f'videos/speed/{FILENAME}' 

# For STREAMING/SOCKET
# DEVICE_ID = 1 # 0 is laptop webcam, 1 is drone
DEVICE_ID = '2024Nov28_resized/DJI_0064_demo.MP4' # Change to video path to debug
if SAVE_MODE == 'VIDEO':
    SAVE_CSV_PATH = f"csv_files/{FILENAME.split('.')[0]}.csv"
    SAVE_VIDEO_PATH = f"saved_annotated_videos/{FILENAME.split('.')[0]}_output.mp4"
elif SAVE_MODE in ['STREAMING', 'SOCKET']:
    from datetime import datetime
    current_date_time = datetime.today().strftime('%Y%m%d_%H%M%S')
    SAVE_CSV_PATH = f'csv_files/{current_date_time}.csv'
    SAVE_VIDEO_PATH = f'saved_stream_videos/{current_date_time}_output.mp4'


# Convert pixels to meters
D = 8 # drone height is 8m
SENSOR_SIZE = (17.3/1000, 13.3/1000) # 17.3 x 13 mm
F = 24/1000 # focal length 24mm
# F = 10.26/1000 # actual focal length 10.26mm
RESOLUTION = (900, 600) # resize of 5472, 3648
PIXEL_DENSITY_WIDTH = RESOLUTION[0]/SENSOR_SIZE[0]
PIXEL_DENSITY_HEIGHT = RESOLUTION[1]/SENSOR_SIZE[1]
REAL_SIZE_WIDTH = D/(F*PIXEL_DENSITY_WIDTH) # size of 1 pixel in meter
REAL_SIZE_HEIGHT = D/(F*PIXEL_DENSITY_HEIGHT) # size of 1 pixel in meter
