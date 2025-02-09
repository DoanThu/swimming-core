from config.general import MODE, SAVE_AFTER_SECONDS, SAVE_JSON_PATH, SAVE_CSV_PATH, VIDEO_PATH, DEVICE_ID, SAVE_VIDEO_PATH, DEFAULT_REDIS_HOST, DEFAULT_REDIS_PORT, FPS_RATE
import cv2 
from utils.file_utils import write_to_csv, write_json
from postprocess.data import FrameData
from typing import List
from main_process.main_processing import MainCalculation
import os
import sys
import redis
import time
from utils.dict_utils import get_first_k
from postprocess.analytics import ExtractParams
from db_process.redis_process import image_to_redis, frame_data_to_redis, dict_to_redis
import logging 
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')

frame_data_list: List[FrameData] = []

def run_video(debug=False, save_json=False, save_csv=False, out_video=SAVE_VIDEO_PATH, fx=1, fy=1):
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_width, frame_height = int(cap.get(3)*fx), int(cap.get(4)*fy)
    if frame_width == 0 and frame_height == 0:
        logging.info(f'Cannot find video at {VIDEO_PATH}.')
        return 
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_idx = 0
    output = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'MP4V'),
                             fps, (frame_width, frame_height))
    
    logging.info(f'frame_width={frame_width}, frame_height={frame_width}, fps = {fps}')

    if save_csv:
        if os.path.exists(SAVE_CSV_PATH):
            os.remove(SAVE_CSV_PATH)
    
    calculate_frame = MainCalculation(fps)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            out_frame, frame_data, _ = calculate_frame.swimming_calculation(frame=frame, frame_idx=frame_idx, debug=debug)
            frame_idx += 1
            
            # save SAVE_AFTER_SECONDS frames in json format
            if frame_idx % (SAVE_AFTER_SECONDS*fps) == 0:
                if save_json:
                    filename = SAVE_JSON_PATH.format(frame_idx//(SAVE_AFTER_SECONDS*fps))
                    save_json([_frame.__dict__ for _frame in frame_data_list[frame_idx-(SAVE_AFTER_SECONDS*fps):]], filename)
                    logging.info(f'Saved json file to {filename}')
                else:
                    logging.debug('>>>>> {} seconds elapsed'.format(SAVE_AFTER_SECONDS))
                    
            if save_csv:
                data = str(frame_data.red_marker) + ',' + str(frame_data.direction) + ',' + str(frame_data.speed) + ',' + ','.join(str(p) for p in frame_data.skeleton)
                write_to_csv(data, SAVE_CSV_PATH)
            
            if out_video:
                output.write(out_frame)
            
        else:
            break
        
    cap.release()
    output.release()
    
    if debug:
        logging.debug(f'Annotated video is saved at {out_video}')
    if save_csv:
        logging.debug(f'Csv file is saved at {SAVE_CSV_PATH}')


def run_stream(debug=False, save_json=False, save_csv=False, out_video=SAVE_VIDEO_PATH, fx=1, fy=1):
    # Connect to Redis
    r = redis.Redis(host=DEFAULT_REDIS_HOST, port=DEFAULT_REDIS_PORT, db=0)


    cap = cv2.VideoCapture(DEVICE_ID)
    frame_width, frame_height = int(cap.get(3)*fx), int(cap.get(4)*fy)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_idx = -1
    output = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'MP4V'),
                             fps//FPS_RATE, (frame_width, frame_height))
    
    logging.info(f'frame_width={frame_width}, frame_height={frame_width}, fps = {fps}')

    if save_csv:
        if os.path.exists(SAVE_CSV_PATH):
            os.remove(SAVE_CSV_PATH)

    if not os.path.exists(os.path.dirname(out_video)):
        os.makedirs(os.path.dirname(out_video))
    
    calculate_frame = MainCalculation(fps)
    extractParams = ExtractParams()

    prev_features_list = []
    cur_features_list = []

    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (0, 0), fx = fx, fy = fy)
                frame_idx += 1
                if frame_idx % (SAVE_AFTER_SECONDS*fps) == 0:
                        logging.debug(f'>>>>> {SAVE_AFTER_SECONDS} seconds elapsed. Current frame idx is {frame_idx}')
                if frame_idx % FPS_RATE != 0: 
                    continue
                   
                
                annotated_frame, frame_data, updated_speed = calculate_frame.swimming_calculation(frame=frame, frame_idx=frame_idx, debug=debug)
                
                # save SAVE_AFTER_SECONDS frames in json format
                # if frame_idx % (SAVE_AFTER_SECONDS*fps) == 0:
                    # if save_json:
                        # filename = SAVE_JSON_PATH.format(frame_idx//(SAVE_AFTER_SECONDS*fps))
                        # write_json([_frame.__dict__ for _frame in frame_data_list[frame_idx-(SAVE_AFTER_SECONDS*fps):]], filename)
                        # logging.info(f'Saved json file to {filename}')

                
                
                
                image_to_redis(r, frame, 'raw_image')
                image_to_redis(r, annotated_frame, 'annotated_image')
                frame_data_to_redis(r, frame_data, 'frame_data')

                if not frame_data.skeleton: continue

                if updated_speed:
                    if not prev_features_list:
                        prev_features_list = cur_features_list
                        cur_features_list = []
                    else:
                        d_distances = extractParams.extract_distance_features(frame_data.skeleton)
                        d_angles = extractParams.extract_angle_features(frame_data.skeleton)
                        cur_features_list.append([d_distances, d_angles])
                        extractParams.extract_pct_change_list(prev_features_list, cur_features_list)

                        dist_feature_list = ['nose_lwrist', 'nose_rwrist', 'lwrist_rwrist']
                        dict_to_redis(r, {feature:extractParams.cur_dist_features[feature] for feature in dist_feature_list}, 'cur_dist_features')

                        angle_feature_list = ['lshoulder_lelbow_lwrist', 'rshoulder_relbow_rwrist']
                        dict_to_redis(r, {feature:extractParams.cur_angle_features[feature] for feature in angle_feature_list}, 'cur_angle_features')

                        if extractParams.pct_dist_changes: # not empty
                            dict_to_redis(r,extractParams.pct_dist_changes,'pct_dist_changes')
                        if extractParams.pct_angle_changes: # not empty
                            dict_to_redis(r,extractParams.pct_angle_changes,'pct_angle_changes')

                        if save_csv:
                            data = str(frame_data.direction) + ',' + str(frame_data.speed) + ',' + str(frame_data.speed_pct_change)+ ',' + str(get_first_k(extractParams.pct_dist_changes, 5)) + ',' + str(get_first_k(extractParams.pct_angle_changes, 5))
                            write_to_csv(data, SAVE_CSV_PATH)
                else:
                    d_distances = extractParams.extract_distance_features(frame_data.skeleton)
                    d_angles = extractParams.extract_angle_features(frame_data.skeleton)
                    cur_features_list.append([d_distances, d_angles])

                
                
                if out_video:
                    output.write(annotated_frame)
                
            else:
                if save_csv:
                    logging.info(f'Saved csv file to {SAVE_CSV_PATH}')

                cap.release()
                output.release()
        except KeyboardInterrupt:
            cap.release()
            output.release()
            if save_csv:
                logging.info(f'Saved csv file to {SAVE_CSV_PATH}')
            logging.info(f'Saved video to {out_video}')
            sys.exit()
    
    if debug:
        logging.debug(f'Annotated video is saved at {out_video}')
    if save_csv:
        logging.debug(f'Csv file is saved at {SAVE_CSV_PATH}')



if __name__ == '__main__':
    if MODE == 'VIDEO':
        run_video(debug=False, save_json=False, save_csv=True)
    elif MODE == 'STREAMING':
        run_stream(debug=False, save_json=False, save_csv=True, fx=0.4, fy=0.4)
    
    
   