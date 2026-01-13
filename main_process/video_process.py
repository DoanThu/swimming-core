from config.general import SAVE_AFTER_SECONDS, SAVE_CSV_PATH, SAVE_VIDEO_PATH, FPS_RATE, SAVE_CSV_COLUMNS, SAVE_SKELETON_PATH, SAVE_ANALYSIS_PATH_SINGLE, SAVE_ANALYSIS_PATH_MULTI, POSE_CONFIG, SEG_CONFIG
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
import queue
from model_caller.seg_model_caller import SegWorker
from model_caller.pose_model_caller import PoseWorker
from utils.file_utils import read_yaml
import torch


def run_video(video_path, debug=False, save_csv=False, out_video=None, fx=1, fy=1, lane_type='segmentation'):
    from main_process.core_process_single import MainCalculation
    cap = cv2.VideoCapture(video_path)
    if fx < 1 and fy < 1:
        frame_width, frame_height = int(cap.get(3)*fx), int(cap.get(4)*fy)
    else:
        frame_width, frame_height = fx, fy

    fps = round(cap.get(cv2.CAP_PROP_FPS))
    frame_idx = -1

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    if out_video is None:
        if not os.path.exists('saved_annotated_videos'):
            os.makedirs('saved_annotated_videos')
        out_video = f"saved_annotated_videos/{base_name}_output.mp4"

    if not os.path.exists(os.path.dirname(out_video)):
        os.makedirs(os.path.dirname(out_video))
    if os.path.exists(out_video):
        os.remove(out_video)
    output = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'MP4V'),
                             fps//FPS_RATE, (frame_width, frame_height))
    
    logging.info(f'frame_width={frame_width}, frame_height={frame_height}, fps = {fps}')

    if save_csv:
        if not os.path.exists('csv_files'):
            os.makedirs('csv_files')
        save_csv_path = f"csv_files/{base_name}.csv"
        if os.path.exists(save_csv_path):
            os.remove(save_csv_path)
        write_to_csv(','.join(SAVE_CSV_COLUMNS), save_csv_path)
    
    
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
                    logging.info(f"Saved csv file to {save_csv_path}")
                cap.release()
                output.release()
                logging.info(f'Saved video to {out_video}')

        except KeyboardInterrupt:
            cap.release()
            output.release()
            if save_csv:
                logging.info(f"Saved csv file to {save_csv_path}")
            logging.info(f'Saved video to {out_video}')
            sys.exit()
    

    filename = os.path.splitext(os.path.basename(out_video))[0]
    save_analysis_path = f'{SAVE_ANALYSIS_PATH_SINGLE}/{filename}'
    frame_list = [(i,f) for i,f in enumerate(frame_list)]
    save_video_and_index(frame_list, frame_data_list, out_dir=save_analysis_path, fps=fps//FPS_RATE)
    logging.info(f'Saved analysis to {save_analysis_path}.')

   
def run_video_multi(video_path, debug=False, save_csv=False, out_video=None, fx=1, fy=1):
    from main_process.core_process_multiple import MainCalculation

    cap = cv2.VideoCapture(video_path)
    if fx < 1 and fy < 1:
        frame_width, frame_height = int(cap.get(3)*fx), int(cap.get(4)*fy)
    else:
        frame_width, frame_height = fx, fy

    fps = round(cap.get(cv2.CAP_PROP_FPS))
    frame_idx = -1

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    if out_video is None:
        if not os.path.exists('saved_annotated_videos'):
            os.makedirs('saved_annotated_videos')
        out_video = f"saved_annotated_videos/{base_name}_output.mp4"

    if not os.path.exists(os.path.dirname(out_video)):
        os.makedirs(os.path.dirname(out_video))
    if os.path.exists(out_video):
        os.remove(out_video)
    output = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'MP4V'),
                             fps//FPS_RATE, (frame_width, frame_height))
    
    logging.info(f'frame_width={frame_width}, frame_height={frame_height}, fps = {fps}')

    if save_csv:
        if not os.path.exists('csv_files'):
            os.makedirs('csv_files')
        save_csv_path = f"csv_files/{base_name}.csv"
        if os.path.exists(save_csv_path):
            os.remove(save_csv_path)
        write_to_csv(','.join(SAVE_CSV_COLUMNS), save_csv_path)
    
    
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
                #             write_to_csv(data, save_csv_path)
                        
                #         prev_features_list = cur_features_list
                #         cur_features_list = []
                # else:
                #     d_distances = extractParams.extract_distance_features(frame_data.skeleton)
                #     d_angles = extractParams.extract_angle_features(frame_data.skeleton)
                #     cur_features_list.append([d_distances, d_angles])
                
                
                
            else:
                cap.release()
                output.release()
                if save_csv:
                    logging.info(f"Saved csv file to {save_csv_path}")
                logging.info(f'Saved video to {out_video}')

        except KeyboardInterrupt:
            cap.release()
            output.release()
            if save_csv:
                logging.info(f"Saved csv file to {save_csv_path}")
            logging.info(f'Saved video to {out_video}')
            sys.exit()
    


    filename = os.path.splitext(os.path.basename(out_video))[0]
    save_analysis_path = f'{SAVE_ANALYSIS_PATH_MULTI}/{filename}'
    frame_list = [(i,f) for i,f in enumerate(frame_list)]
    save_video_and_index(frame_list, frame_data_list, out_dir=save_analysis_path, fps=fps//FPS_RATE)
    logging.info(f'Saved analysis to {save_analysis_path}.')


def run_video_multi_threaded(video_path, debug=False, save_csv=False, out_video=None, fx=1, fy=1):
    from main_process.core_process_multiple_threaded import MainCalculation

    # Runtime knobs to reduce CPU overhead
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    cv2.setNumThreads(0)
    

    cap = cv2.VideoCapture(video_path)
    if fx < 1 and fy < 1:
        frame_width, frame_height = int(cap.get(3)*fx), int(cap.get(4)*fy)
    else:
        frame_width, frame_height = fx, fy

    fps = round(cap.get(cv2.CAP_PROP_FPS))
    frame_idx = -2 # start at -1 because we skip first frame for processing

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    if out_video is None:
        if not os.path.exists('saved_annotated_videos'):
            os.makedirs('saved_annotated_videos')
        out_video = f"saved_annotated_videos/{base_name}_output.mp4"

    if not os.path.exists(os.path.dirname(out_video)):
        os.makedirs(os.path.dirname(out_video))
    if os.path.exists(out_video):
        os.remove(out_video)
    output = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'MP4V'),
                             fps//FPS_RATE, (frame_width, frame_height))
    
    logging.info(f'frame_width={frame_width}, frame_height={frame_height}, fps = {fps}')

    
    calculate_frame = MainCalculation(fps)
    extractParams = ExtractParams()

    prev_features_list = []
    cur_features_list = []

    overlap_list = []
    frame_data_list: List[FrameMultipleData] = []
    frame_list: Tuple[int, np.ndarray] = []


    pose_config = read_yaml(POSE_CONFIG)
    seg_config = read_yaml(SEG_CONFIG)
    # Initialize persistent threads
    pose_in_q, pose_out_q = queue.Queue(maxsize=4), queue.Queue()
    seg_in_q,  seg_out_q  = queue.Queue(maxsize=4), queue.Queue()

    pose_thread = PoseWorker(pose_config['model_path'], pose_in_q, pose_out_q, **pose_config['inference'])
    seg_thread  = SegWorker(seg_config['model_path'],  seg_in_q,  seg_out_q, **seg_config['inference'])
    
    # Start persistent workers
    pose_thread.start()
    seg_thread.start()
    prev_frame = None

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
                if frame_idx > 0 and frame_idx % FPS_RATE != 0: 
                    continue
                
                init_time = time.perf_counter()        
                
                # Enqueue to BOTH workers
                pose_in_q.put((frame, frame_idx))
                seg_in_q.put((frame, frame_idx))

                # Run previous frame's calculation while waiting for current frame's results
                if frame_idx > -1: # skip first frame processing
                    annotated_frame, frame_data, updated_speed, overlap = calculate_frame.swimming_calculation(frame=prev_frame, 
                                                                                                               frame_idx=frame_idx,
                                                                                                               frame_keypoints=frame_keypoints, 
                                                                                                               lane_divider_bboxes=lane_divider_bboxes,
                                                                                                               debug=debug)
                    # frame_data = FrameMultipleData()
                    # annotated_frame = frame
                
                # Deque from BOTH workers
                frame_idx_p, frame_keypoints, (p0, p1) = pose_out_q.get()
                frame_idx_s, lane_divider_bboxes,  (s0, s1) = seg_out_q.get()

                assert frame_idx_p == frame_idx and frame_idx_s == frame_idx
                if frame_idx > -1:
                    frame_data.pose_time = p1 - p0
                    frame_data.segment_time = s1 - s0
                    
                    frame_data_list.append(frame_data)
                    frame_list.append(frame)
                torch.cuda.synchronize()

                if frame_idx > -1:
                    frame_data.total_time = time.perf_counter() - init_time
                    if out_video:
                        output.write(annotated_frame)

                    if debug:
                        logging.info(f'Pose time: {(p1 - p0)*1000.0:.2f} ms, Seg time: {(s1 - s0)*1000.0:.2f} ms, Total frame time: {frame_data.total_time*1000.0:.2f} ms')

                prev_frame = frame
                    
            else:
                cap.release()
                output.release()
                logging.info(f'Saved video to {out_video}')
                pose_in_q.put(None);  pose_thread.stop_flag.set();  pose_thread.join()
                seg_in_q.put(None);   seg_thread.stop_flag.set();   seg_thread.join()
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass


        except KeyboardInterrupt:
            cap.release()
            output.release()
            logging.info(f'Saved video to {out_video}')
            pose_in_q.put(None);  pose_thread.stop_flag.set();  pose_thread.join()
            seg_in_q.put(None);   seg_thread.stop_flag.set();   seg_thread.join()
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            sys.exit()
    
    if debug:
        total_time_list = [fd.total_time for fd in frame_data_list if fd.total_time is not None]
        logging.info(f'Average processing  time and std dev per frame: {np.mean(total_time_list)*1000.0} {np.std(total_time_list)*1000.0} ms')
        pose_time_list = [fd.pose_time for fd in frame_data_list if fd.pose_time is not None]
        logging.info(f'Average pose model time and std dev per frame: {np.mean(pose_time_list)*1000.0} {np.std(pose_time_list)*1000.0} ms')
        seg_time_list = [fd.segment_time for fd in frame_data_list if fd.segment_time is not None]
        logging.info(f'Average segmentation model time and std dev per frame: {np.mean(seg_time_list)*1000.0} {np.std(seg_time_list)*1000.0} ms')
        analysis_time_list = [fd.analysis_time for fd in frame_data_list if fd.analysis_time is not None]
        logging.info(f'Average analysis time and std dev per frame: {np.mean(analysis_time_list)*1000.0} {np.std(analysis_time_list)*1000.0} ms')

    filename = os.path.splitext(os.path.basename(out_video))[0]
    save_analysis_path = f'{SAVE_ANALYSIS_PATH_MULTI}/{filename}'
    frame_list = [(i,f) for i,f in enumerate(frame_list)]
    save_video_and_index(frame_list, frame_data_list, out_dir=save_analysis_path, fps=fps//FPS_RATE)
    logging.info(f'Saved analysis to {save_analysis_path}.')
