import socket
# Input video has FPS of 60
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
SAVE_CSV_COLUMNS = ['timestamp', 'field', 'value']

# For VIDEO
FILENAME = 'DJI_0306_resized.MP4' # used for multiple swimmers
VIDEO_PATH = f'videos/compare/{FILENAME}' 

# FILENAME = 'DJI_0146.MP4' # used for single swimmer
# VIDEO_PATH = f'videos/speed/{FILENAME}' 

# For STREAMING/SOCKET
# DEVICE_ID = 1 # 0 is laptop webcam, 1 is drone
DEVICE_ID = VIDEO_PATH # Change to video path to debug UI
SERVER_HOST = socket.gethostname()
SERVER_PORT = 9999
from datetime import datetime
current_date_time = datetime.today().strftime('%Y%m%d_%H%M%S')

# Export paths
SAVE_CSV_PATH = {'VIDEO':f"csv_files/{FILENAME.split('.')[0]}.csv",
                'SOCKET': f'csv_files/{current_date_time}.csv'
                }
SAVE_VIDEO_PATH = {'VIDEO': f"saved_annotated_videos/{FILENAME.split('.')[0]}_output.mp4",
                    'SOCKET': f'saved_stream_videos/{current_date_time}_output.mp4'
                    }
SAVE_SKELETON_PATH = 'saved_skeletons/'
SAVE_ANALYSIS_PATH_SINGLE = 'saved_analysis/single'
SAVE_ANALYSIS_PATH_MULTI = 'saved_analysis/multi'


# Convert pixels to meters
D = 8 # drone height is 8m
SENSOR_SIZE = (17.3/1000, 13.3/1000) # 17.3 x 13 mm
F = 24/1000 # focal length 24mm
# RESOLUTION = (900, 600) # resize of 5472, 3648
RESOLUTION = (600, 400) # resize of 5472, 3648
PIXEL_DENSITY_WIDTH = RESOLUTION[0]/SENSOR_SIZE[0]
PIXEL_DENSITY_HEIGHT = RESOLUTION[1]/SENSOR_SIZE[1]
REAL_SIZE_WIDTH = D/(F*PIXEL_DENSITY_WIDTH) # size of 1 pixel in meter
REAL_SIZE_HEIGHT = D/(F*PIXEL_DENSITY_HEIGHT) # size of 1 pixel in meter


# Computation
TIME_WINDOW_STROKE = 150 # 5 seconds for 30 FPS video
ID_TRACKER_WINDOW = 30 # 1 second