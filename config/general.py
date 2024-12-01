# Model
POSE_CONFIG = 'config/pose_model.yaml'
SEG_CONFIG = 'config/seg_model.yaml'
FACE_CONFIG = 'config/face_model.yaml'

# Sytem 
SAVE_AFTER_FRAMES = 60*5 # 60 is fps, 5 is seconds
SAVE_JSON_PATH = 'frame_info/interval_{}.json'


FILENAME = 'DJI_0057.MP4'
VIDEO_PATH = f'2024Nov28_resized/{FILENAME}' 
SWIMMER_POSITION_X = 700 
SWIMMER_POSITION_Y = 300 
SAVE_CSV_PATH = f'csv_files/{FILENAME.split('.')[0]}.csv'
