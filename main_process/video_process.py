from config.general import SAVE_AFTER_SECONDS, SAVE_CSV_PATH, VIDEO_PATH, SAVE_VIDEO_PATH, FPS_RATE, SAVE_CSV_COLUMNS, SAVE_SKELETON_PATH, SAVE_ANALYSIS_PATH_SINGLE, SAVE_ANALYSIS_PATH_MULTI
import cv2 
from utils.file_utils import write_to_csv
from postprocess.single_data import FrameData
from postprocess.multiple_data import FrameMultipleData
from typing import List, Tuple
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
from utils.export_utils import save_skeleton_rows
import numpy as np
from utils.save_processed_info import save_video_and_index



def run_video(debug=False, save_csv=False, out_video=SAVE_VIDEO_PATH['VIDEO'], fx=1, fy=1, lane_type='segmentation'):
    from main_process.core_process_single import MainCalculation
    cap = cv2.VideoCapture(VIDEO_PATH)
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
        if os.path.exists(SAVE_CSV_PATH['VIDEO']):
            os.remove(SAVE_CSV_PATH['VIDEO'])
        write_to_csv(','.join(SAVE_CSV_COLUMNS), SAVE_CSV_PATH['VIDEO'])
        
    save_csv_path = SAVE_CSV_PATH['VIDEO']
    
    
    calculate_frame = MainCalculation(fps)
    extractParams = ExtractParams()

    prev_features_list = []
    cur_features_list = []

    overlap_list = []
    frame_data_list: List[FrameData] = []
    frame_list: Tuple[int, np.ndarray] = []


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
                frame_list.append(frame)

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
                            data =  second_to_time_str(frame_idx/fps) + ',' + str(frame_data.speed_m) + ',' + str(frame_data.speed_pct_change)+ ',' + str(dict_to_string(extractParams.pct_dist_changes, 5)) + ',' + str(dict_to_string(extractParams.pct_angle_changes, 5))
                            write_to_csv(data, SAVE_CSV_PATH['VIDEO'])
                        
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
                    logging.info(f"Saved csv file to {SAVE_CSV_PATH['VIDEO']}")
                cap.release()
                output.release()
                logging.info(f'Saved video to {out_video}')

        except KeyboardInterrupt:
            cap.release()
            output.release()
            if save_csv:
                logging.info(f"Saved csv file to {SAVE_CSV_PATH['VIDEO']}")
            logging.info(f'Saved video to {out_video}')
            sys.exit()
    

    filename = out_video.replace('.mp4', '').split('/')[-1]
    save_analysis_path = f'{SAVE_ANALYSIS_PATH_SINGLE}/{filename}'
    frame_list = [(i,f) for i,f in enumerate(frame_list)]
    save_video_and_index(frame_list, frame_data_list, out_dir=save_analysis_path, fps=fps//FPS_RATE)
    logging.info(f'Saved analysis to {save_analysis_path}.')

    # # Save skeletons
    # os.makedirs('saved_skeletons', exist_ok=True)
    # skeleton_file_name = 'saved_skeletons' + '/' + out_video.replace('.mp4', '_skeletons.txt').split('/')[-1]
    # save_skeleton_rows([s.skeleton[0] for s in frame_data_list],
    #                     out_path=skeleton_file_name,
    #                     tail=[3]*len(frame_data_list)) # tail=5 means breaststroke, 3 means freestyle
    # logging.info(f'Saved skeletons to {skeleton_file_name}')

    # # Lap time
    # laptime_process = LapTime()
    # overlap_list = laptime_process.remove_short_peaks(overlap_list)
    # laptimes = laptime_process.find_zero_segments(overlap_list, fps=fps//2)
    # laptimes_arr = []
    # for i,r in enumerate(laptimes):
    #     if i == 0:
    #         start_ = r[0]
    #     else:
    #         start_ = laptimes[i-1][1]
    #     end_ = r[1]
    #     laptimes_arr.append(start_)
    #     laptimes_arr.append(end_)
    # if save_csv:
    #     laptimes_arr_str = [str(i) for i in laptimes_arr]
    #     data =  '-1' + ',' + 'lap_times' + ',' + ','.join(laptimes_arr_str)
    #     write_to_csv(data, save_csv_path)

    # Stroke count
    # from model_caller.stroke_classification_caller import StrokeClassificationCaller
    # stroke_classification_caller = StrokeClassificationCaller()
    # stroke_count_list = []
    # for i in range(0,len(laptimes_arr),2):
    #     start_idx = int(laptimes_arr[i]*30)
    #     end_idx = int(laptimes_arr[i+1]*30)
    #     print(f'start_idx={start_idx}, end_idx={end_idx}')
    #     skeletons = [frame.skeleton[0] for frame in frame_data_list[start_idx:end_idx]]
    #     stroke_type = stroke_classification_caller.inference(skeletons)
    #     skeletons = flip_skeletons(skeletons)
    #     nose_wrist = get_nose_wrist_distance(skeletons)
    #     nose_wrist = moving_average(nose_wrist)
    #     stroke_count = get_stroke_count_cap(nose_wrist)
    #     if stroke_type in [1,2]:
    #         stroke_count *= 2
    #     stroke_count_list.append(stroke_count)

    # if save_csv:
    #     stroke_count_list_str = [str(i) for i in stroke_count_list] 
    #     data =  '-1' + ',' + 'stroke_count' + ',' + ','.join(stroke_count_list_str)
    #     write_to_csv(data, save_csv_path)
    
    # if save_csv:
    #     logging.info(f'Saved csv file to {save_csv_path}')
   
def run_video_multi(debug=False, save_csv=False, out_video=SAVE_VIDEO_PATH['VIDEO'], fx=1, fy=1):
    from main_process.core_process_multiple import MainCalculation

    cap = cv2.VideoCapture(VIDEO_PATH)
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
        if os.path.exists(SAVE_CSV_PATH['VIDEO']):
            os.remove(SAVE_CSV_PATH['VIDEO'])
        write_to_csv(','.join(SAVE_CSV_COLUMNS), SAVE_CSV_PATH['VIDEO'])
        
    save_csv_path = SAVE_CSV_PATH['VIDEO']
    
    
    calculate_frame = MainCalculation(fps)
    extractParams = ExtractParams()

    prev_features_list = []
    cur_features_list = []

    overlap_list = []
    frame_data_list: List[FrameMultipleData] = []
    frame_list: Tuple[int, np.ndarray] = []


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
                
                if out_video:
                    output.write(annotated_frame)
                
                frame_data_list.append(frame_data)
                frame_list.append(frame)
                    
                if not frame_data.skeleton_list: continue
                

                # overlap_list.append(overlap) # unused so far

                # if updated_speed:
                #     if not prev_features_list:
                #         prev_features_list = cur_features_list
                #         cur_features_list = []
                #     else:
                #         d_distances = extractParams.extract_distance_features(frame_data.skeleton)
                #         d_angles = extractParams.extract_angle_features(frame_data.skeleton)
                #         cur_features_list.append([d_distances, d_angles])
                #         extractParams.extract_pct_change_list(prev_features_list, cur_features_list)


                #         if save_csv:
                #             data =  second_to_time_str(frame_idx/fps) + ',' + str(frame_data.speed_m) + ',' + str(frame_data.speed_pct_change)+ ',' + str(dict_to_string(extractParams.pct_dist_changes, 5)) + ',' + str(dict_to_string(extractParams.pct_angle_changes, 5))
                #             write_to_csv(data, SAVE_CSV_PATH['VIDEO'])
                        
                #         prev_features_list = cur_features_list
                #         cur_features_list = []
                # else:
                #     d_distances = extractParams.extract_distance_features(frame_data.skeleton)
                #     d_angles = extractParams.extract_angle_features(frame_data.skeleton)
                #     cur_features_list.append([d_distances, d_angles])
                
                
                
            else:
                if save_csv:
                    logging.info(f"Saved csv file to {SAVE_CSV_PATH['VIDEO']}")
                cap.release()
                output.release()
                logging.info(f'Saved video to {out_video}')

        except KeyboardInterrupt:
            cap.release()
            output.release()
            if save_csv:
                logging.info(f"Saved csv file to {SAVE_CSV_PATH['VIDEO']}")
            logging.info(f'Saved video to {out_video}')
            sys.exit()
    


    filename = out_video.replace('.mp4', '').split('/')[-1]
    save_analysis_path = f'{SAVE_ANALYSIS_PATH_MULTI}/{filename}'
    frame_list = [(i,f) for i,f in enumerate(frame_list)]
    save_video_and_index(frame_list, frame_data_list, out_dir=save_analysis_path, fps=fps//FPS_RATE)
    logging.info(f'Saved analysis to {save_analysis_path}.')


