MODE = 'STREAMING' # MODE is VIDEO or STREAMING
FPS_RATE = 2 # read every n frames, needs to be a factor of with freq_frame_idx in seg_model.yaml

# Model
POSE_CONFIG = 'config/pose_model.yaml'
SEG_CONFIG = 'config/seg_model.yaml'
FACE_CONFIG = 'config/face_model.yaml'

# Sytem 
SAVE_AFTER_SECONDS = 5 
SAVE_JSON_PATH = 'frame_info/interval_{}.json'

# For VIDEO
FILENAME = 'DJI_0057.MP4'
VIDEO_PATH = f'2024Nov28_resized/{FILENAME}' 
# For STREAMING
# DEVICE_ID = 0  
DEVICE_ID = '2024Nov28_resized/DJI_0057.MP4' # Change to video path to debug
if MODE == 'VIDEO':
    SAVE_CSV_PATH = f'csv_files/{FILENAME.split('.')[0]}.csv'
    SAVE_VIDEO_PATH = f'{FILENAME.split('.')[0]}_output.mp4'
elif MODE == 'STREAMING':
    from datetime import datetime
    current_date_time = datetime.today().strftime('%Y%m%d_%H%M%S')
    SAVE_CSV_PATH = f'csv_files/{current_date_time}.csv'
    SAVE_VIDEO_PATH = f'saved_stream_videos/{current_date_time}_output.mp4'


# Redis for STREAMING
DEFAULT_REDIS_PORT = 6379
DEFAULT_REDIS_HOST = 'localhost'