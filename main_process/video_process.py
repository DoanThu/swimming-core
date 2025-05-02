from config.general import SAVE_AFTER_SECONDS, VIDEO_PATH, SAVE_VIDEO_PATH, FPS_RATE, SAVE_CSV_COLUMNS
import cv2 
from utils.file_utils import write_to_csv, write_json
from postprocess.data import FrameData
from typing import List
from main_process.core_process import MainCalculation
import os
import sys
import time
from postprocess.lap_time_process import LapTime
from utils.time_utils import second_to_time_str
from utils.dict_utils import dict_to_string
from postprocess.analytics import ExtractParams
import logging 
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')
from utils.signal_utils import moving_average
from postprocess.stroke_count_process import get_stroke_count_cap, flip_skeletons, get_nose_wrist_distance

def run_video(debug=False, video_path=VIDEO_PATH, save_csv=False, out_video=SAVE_VIDEO_PATH, fx=1, fy=1):
    cap = cv2.VideoCapture(video_path)
    if fx < 1 and fy < 1:
        frame_width, frame_height = int(cap.get(3)*fx), int(cap.get(4)*fy)
    else:
        frame_width, frame_height = fx, fy

    fps = round(cap.get(cv2.CAP_PROP_FPS))
    frame_idx = -1

    if not os.path.exists(os.path.dirname(out_video)):
        os.makedirs(os.path.dirname(out_video))
    output = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'MP4V'),
                             fps//FPS_RATE, (frame_width, frame_height))
    
    logging.info(f'frame_width={frame_width}, frame_height={frame_height}, fps = {fps}')

    if save_csv:
        filename = video_path.split('/')[-1].split('.')[0]
        print(video_path)
        save_csv_path = f"csv_files/{filename}.csv"
        if os.path.exists(save_csv_path):
            os.remove(save_csv_path)
        write_to_csv(','.join(SAVE_CSV_COLUMNS), save_csv_path)
        

    
    
    calculate_frame = MainCalculation(fps)
    extractParams = ExtractParams()

    prev_features_list = []
    cur_features_list = []

    overlap_list = []
    frame_data_list: List[FrameData] = []


    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if ret:
                if fx < 1 and fy < 1:
                    frame = cv2.resize(frame, (0, 0), fx=fx, fy=fy)
                else:
                    frame = cv2.resize(frame, (fx, fy)) 

                frame_idx += 1
                if frame_idx % (SAVE_AFTER_SECONDS*fps) == 0:
                        logging.debug(f'>>>>> {SAVE_AFTER_SECONDS} seconds elapsed. Current frame idx is {frame_idx}')
                if frame_idx % FPS_RATE != 0: 
                    continue
                   
                
                annotated_frame, frame_data, updated_speed, overlap = calculate_frame.swimming_calculation(frame=frame, frame_idx=frame_idx, debug=debug)
                
                if not frame_data.skeleton: continue

                overlap_list.append(overlap)
                frame_data_list.append(frame_data)

                if updated_speed:
                    if not prev_features_list:
                        prev_features_list = cur_features_list
                        cur_features_list = []
                    else:
                        d_distances = extractParams.extract_distance_features(frame_data.skeleton)
                        d_angles = extractParams.extract_angle_features(frame_data.skeleton)
                        cur_features_list.append([d_distances, d_angles])
                        extractParams.extract_pct_change_list(prev_features_list, cur_features_list)


                        if save_csv:
                            # data =  second_to_time_str(frame_idx/fps) + ',' + str(frame_data.speed_m) + ',' + str(frame_data.speed_pct_change)+ ',' + str(dict_to_string(extractParams.pct_dist_changes, 5)) + ',' + str(dict_to_string(extractParams.pct_angle_changes, 5))
                            data =  second_to_time_str(frame_idx/fps) + ',' + 'speed' + ',' + str(frame_data.speed_m) 
                            write_to_csv(data, save_csv_path)
                        
                        prev_features_list = cur_features_list
                        cur_features_list = []
                else:
                    d_distances = extractParams.extract_distance_features(frame_data.skeleton)
                    d_angles = extractParams.extract_angle_features(frame_data.skeleton)
                    cur_features_list.append([d_distances, d_angles])
                
                
                if out_video:
                    output.write(annotated_frame)
                
            else:
                if save_csv:
                    logging.info(f'Saved csv file to {save_csv_path}')

                cap.release()
                output.release()
        except KeyboardInterrupt:
            cap.release()
            output.release()
            # if save_csv:
                # logging.info(f'Saved csv file to {SAVE_CSV_PATH}')
            logging.info(f'Saved video to {out_video}')
            sys.exit()
    
    if debug:
        logging.debug(f'Annotated video is saved at {out_video}')

    # Lap time
    laptime_process = LapTime()
    overlap_list = laptime_process.remove_short_peaks(overlap_list)
    laptimes = laptime_process.find_zero_segments(overlap_list, fps=fps//2)
    laptimes_arr = []
    for i,r in enumerate(laptimes):
        if i == 0:
            start_ = r[0]
        else:
            start_ = laptimes[i-1][1]
        end_ = r[1]
        laptimes_arr.append(start_)
        laptimes_arr.append(end_)
    if save_csv:
        laptimes_arr_str = [str(i) for i in laptimes_arr]
        data =  '-1' + ',' + 'lap_times' + ',' + ','.join(laptimes_arr_str)
        write_to_csv(data, save_csv_path)

    # Stroke count
    from model_caller.stroke_classification_caller import StrokeClassificationCaller
    stroke_classification_caller = StrokeClassificationCaller()
    stroke_count_list = []
    for i in range(0,len(laptimes_arr),2):
        start_idx = int(laptimes_arr[i]*30)
        end_idx = int(laptimes_arr[i+1]*30)
        print(start_idx, end_idx)
        skeletons = [frame.skeleton[0] for frame in frame_data_list[start_idx:end_idx]]
        stroke_type = stroke_classification_caller.inference(skeletons)
        skeletons = flip_skeletons(skeletons)
        nose_wrist = get_nose_wrist_distance(skeletons)
        nose_wrist = moving_average(nose_wrist)
        stroke_count = get_stroke_count_cap(nose_wrist)
        if stroke_type in [1,2]:
            stroke_count *= 2
        stroke_count_list.append(stroke_count)

    if save_csv:
        stroke_count_list_str = [str(i) for i in stroke_count_list] 
        data =  '-1' + ',' + 'stroke_count' + ',' + ','.join(stroke_count_list_str)
        write_to_csv(data, save_csv_path)
    
    if save_csv:
        logging.info(f'Saved csv file to {save_csv_path}')
   



