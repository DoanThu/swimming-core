MODE = 'STREAMING' # MODE is VIDEO or STREAMING
# Model
POSE_CONFIG = 'config/pose_model.yaml'
SEG_CONFIG = 'config/seg_model.yaml'
FACE_CONFIG = 'config/face_model.yaml'

# Sytem 
SAVE_AFTER_SECONDS = 5 
SAVE_JSON_PATH = 'frame_info/interval_{}.json'

# For VIDEO
FILENAME = 'DJI_0965.MP4'
VIDEO_PATH = f'2024Mar/{FILENAME}' 
# For STREAMING
DEVICE_ID = 0  
if MODE == 'VIDEO':
    SAVE_CSV_PATH = f'csv_files/{FILENAME.split('.')[0]}.csv'
    SAVE_VIDEO_PATH = f'{FILENAME.split('.')[0]}_output.mp4'
elif MODE == 'STREAMING':
    from datetime import datetime
    current_date_time = datetime.today().strftime('%Y%m%d_%H%M%S')
    SAVE_CSV_PATH = f'csv_files/{current_date_time}.csv'
    SAVE_VIDEO_PATH = f'{current_date_time}_output.mp4'
