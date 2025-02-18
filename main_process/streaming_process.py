# from config.general import SAVE_AFTER_SECONDS, SAVE_JSON_PATH, SAVE_CSV_PATH, VIDEO_PATH, DEVICE_ID, SAVE_VIDEO_PATH, DEFAULT_REDIS_HOST, DEFAULT_REDIS_PORT, FPS_RATE, SAVE_CSV_COLUMNS
# import cv2 
# from utils.file_utils import write_to_csv, write_json
# from postprocess.data import FrameData
# from typing import List
# from main_process.core_process import MainCalculation
# import os
# import sys
# import redis
# import time
# from utils.time_utils import second_to_time_str
# from utils.dict_utils import dict_to_string, get_first_k
# from postprocess.analytics import ExtractParams
# from db_process.redis_process import image_to_redis, frame_data_to_redis, dict_to_redis
# import logging 
# logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
#                     level=logging.DEBUG,
#                     datefmt='%Y-%m-%d %H:%M:%S')

# frame_data_list: List[FrameData] = []

# def redis_visualization(r, d_distances, d_angles):
#     # For visualization in client side
#     dist_feature_list = ['nose_lwrist', 'nose_rwrist', 'lwrist_rwrist']
#     dict_to_redis(r, get_first_k({feature:d_distances[feature] for feature in dist_feature_list},5), 'cur_dist_features')

#     angle_feature_list = ['lshoulder_lelbow_lwrist', 'rshoulder_relbow_rwrist']
#     dict_to_redis(r, get_first_k({feature:d_angles[feature] for feature in angle_feature_list},5), 'cur_angle_features')



# def run_stream(debug=False, save_json=False, save_csv=False, out_video=SAVE_VIDEO_PATH, fx=1, fy=1, lane_type='segment'):
#     # Connect to Redis
#     r = redis.Redis(host=DEFAULT_REDIS_HOST, port=DEFAULT_REDIS_PORT, db=0)


#     cap = cv2.VideoCapture(DEVICE_ID)
#     frame_width, frame_height = int(cap.get(3)*fx), int(cap.get(4)*fy)
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     frame_idx = -1
    
#     if not os.path.exists(os.path.dirname(out_video)):
#         os.makedirs(os.path.dirname(out_video))
#     output = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'MP4V'),
#                              fps//FPS_RATE, (frame_width, frame_height))
    
#     logging.info(f'frame_width={frame_width}, frame_height={frame_width}, fps = {fps}')

#     if save_csv:
#         if os.path.exists(SAVE_CSV_PATH):
#             os.remove(SAVE_CSV_PATH)
#         write_to_csv(','.join(SAVE_CSV_COLUMNS), SAVE_CSV_PATH)
        

    
    
#     calculate_frame = MainCalculation(fps, lane_type=lane_type)
#     extractParams = ExtractParams()

#     prev_features_list = []
#     cur_features_list = []

#     while cap.isOpened():
#         try:
#             ret, frame = cap.read()
#             if ret:
#                 frame = cv2.resize(frame, (0, 0), fx = fx, fy = fy)
#                 frame_idx += 1
#                 if frame_idx % (SAVE_AFTER_SECONDS*fps) == 0:
#                         logging.debug(f'>>>>> {SAVE_AFTER_SECONDS} seconds elapsed. Current frame idx is {frame_idx}')
#                 if frame_idx % FPS_RATE != 0: 
#                     continue
                   
                
#                 annotated_frame, frame_data, updated_speed = calculate_frame.swimming_calculation(frame=frame, frame_idx=frame_idx, debug=debug)
                
#                 image_to_redis(r, frame, 'raw_image')
#                 image_to_redis(r, annotated_frame, 'annotated_image')
#                 frame_data_to_redis(r, frame_data, 'frame_data')

#                 if not frame_data.skeleton: continue

#                 if updated_speed:
#                     # Find factors impact speed change
#                     if not prev_features_list:
#                         prev_features_list = cur_features_list
#                         cur_features_list = []
#                     else:
#                         d_distances = extractParams.extract_distance_features(frame_data.skeleton)
#                         d_angles = extractParams.extract_angle_features(frame_data.skeleton)
#                         cur_features_list.append([d_distances, d_angles])
#                         extractParams.extract_pct_change_list(prev_features_list, cur_features_list)


#                         # For visualization in client side
#                         redis_visualization(r, d_distances, d_angles)


#                         # Save which params have changed the most
#                         if extractParams.pct_dist_changes: # not empty
#                             dict_to_redis(r,get_first_k(extractParams.pct_dist_changes,5),'pct_dist_changes')
#                         if extractParams.pct_angle_changes: # not empty
#                             dict_to_redis(r,get_first_k(extractParams.pct_angle_changes,5),'pct_angle_changes')

#                         if save_csv:
#                             data =  second_to_time_str(frame_idx/(fps//FPS_RATE)) + ',' + str(frame_data.speed) + ',' + str(frame_data.speed_pct_change)+ ',' + str(dict_to_string(extractParams.pct_dist_changes, 5)) + ',' + str(dict_to_string(extractParams.pct_angle_changes, 5))
#                             write_to_csv(data, SAVE_CSV_PATH)

#                         prev_features_list = cur_features_list
#                         cur_features_list = []
                    

#                 else:
#                     d_distances = extractParams.extract_distance_features(frame_data.skeleton)
#                     d_angles = extractParams.extract_angle_features(frame_data.skeleton)
#                     cur_features_list.append([d_distances, d_angles])
#                     redis_visualization(r, d_distances, d_angles)



                
                
#                 if out_video:
#                     output.write(annotated_frame)
                
#             else:
#                 if save_csv:
#                     logging.info(f'Saved csv file to {SAVE_CSV_PATH}')

#                 cap.release()
#                 output.release()
#         except KeyboardInterrupt:
#             cap.release()
#             output.release()
#             if save_csv:
#                 logging.info(f'Saved csv file to {SAVE_CSV_PATH}')
#             logging.info(f'Saved video to {out_video}')
#             sys.exit()
    
#     if debug:
#         logging.debug(f'Annotated video is saved at {out_video}')
#     if save_csv:
#         logging.debug(f'Csv file is saved at {SAVE_CSV_PATH}')


    
    
   