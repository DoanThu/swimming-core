import socket,cv2,pickle, struct
from config.general import FPS_RATE, DEVICE_ID, SAVE_AFTER_SECONDS, SAVE_VIDEO_PATH, SAVE_CSV_PATH, POSE_CONFIG, SEG_CONFIG, SERVER_HOST, SERVER_PORT, FREQ_SEGMENT
import logging 
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')
from main_process.core_process_single import MainCalculation
from postprocess.analytics import ExtractParams
import os
from utils.dict_utils import dict_to_string, get_first_k
import traceback
from utils.time_utils import second_to_time_str
import numpy as np
from utils.file_utils import write_to_csv
from postprocess.single_data import FrameDataConst
from main_process.core_process_multiple_threaded import MainCalculation as MainCalculationMulti
from model_caller.seg_model_caller import SegWorker
from model_caller.pose_model_caller import PoseWorker
from utils.file_utils import read_yaml
import queue
import time
import torch
from postprocess.visualization_process import visualize_swimmer_id


def encode_to_send(data):
    a = pickle.dumps(data)
    message = struct.pack("L", len(a))+a
    return message

def send_to_client(clientsocket, list_data):
    message = encode_to_send(len(list_data)) # means sending that many variables
    for data_to_send in list_data: # client needs to extract by this order
        message += encode_to_send(data_to_send)
    clientsocket.sendall(message)

def param_visualization(d_distances, d_angles):
    # For visualization in client side
    dist_feature_list = ['nose_lwrist', 'nose_rwrist', 'lwrist_rwrist']
    dist_visualization = get_first_k({feature:d_distances[feature] for feature in dist_feature_list},5)

    angle_feature_list = ['lshoulder_lelbow_lwrist', 'rshoulder_relbow_rwrist']
    angle_visualization = get_first_k({feature:d_angles[feature] for feature in angle_feature_list},5)
    return dist_visualization, angle_visualization

def run_socket(debug=False, save_csv=False, out_video=SAVE_VIDEO_PATH['SOCKET'], fx=1.0, fy=1.0):
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind((SERVER_HOST, SERVER_PORT))
    serversocket.listen(5)

    logging.info("SERVER STARTED")


    extractParams = ExtractParams()
    try: 
        while True:
            clientsocket, addr = serversocket.accept()
            try:
                if clientsocket:
                    logging.info("Got a connection from %s" % str(addr))

                    cap = cv2.VideoCapture(DEVICE_ID)
                    if fx < 1 and fy < 1:
                        frame_width, frame_height = int(cap.get(3)*fx), int(cap.get(4)*fy)
                    else:
                        frame_width, frame_height = fx, fy
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    common_fps = [24, 30, 60, 120]
                    fps = common_fps[np.argmin([abs(i-fps) for i in common_fps])]

                    calculate_frame = MainCalculation(fps=fps)

                    if not os.path.exists(os.path.dirname(out_video)):
                        os.makedirs(os.path.dirname(out_video))
                    output = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'MP4V'),
                                            fps//FPS_RATE, (frame_width, frame_height))

                    cur_features_list = []
                    prev_features_list = []
                    frame_idx = -1
                    while(cap.isOpened()):
                        frame_idx += 1

                        ret, frame = cap.read()
                        if ret:
                            if fx < 1 and fy < 1:
                                frame = cv2.resize(frame, (0, 0), fx=fx, fy=fy)
                            else:
                                frame = cv2.resize(frame, (fx, fy)) 

                            if frame_idx % (SAVE_AFTER_SECONDS*fps) == 0:
                                logging.info(f'>>>>> {SAVE_AFTER_SECONDS} seconds elapsed. Current frame idx is {frame_idx}')
                            
                            if frame_idx % FPS_RATE != 0: 
                                continue

                            annotated_frame, frame_data, updated_speed, overlap = calculate_frame.swimming_calculation(frame=frame, frame_idx=frame_idx, debug=debug, from_socket=True)
                            
                            if not frame_data.skeleton: continue

                            swimming_speed = frame_data.speed_m
                            d_distances = extractParams.extract_distance_features(frame_data.skeleton) # single point

                            sub_frame_data = {'speed': swimming_speed, 'pct_change': frame_data.speed_pct_change,
                                               'stroke': FrameDataConst.MAP_STROKE[frame_data.stroke],
                                               'direction': FrameDataConst.MAP_DIRECTION[frame_data.direction],
                                               'stroke_count': frame_data.stroke_count,
                                               'status': FrameDataConst.MAP_STATUS[frame_data.status]
                                               }

                            if not frame_data.skeleton: 
                                send_to_client(clientsocket, [second_to_time_str(frame_idx/fps), annotated_frame, frame, sub_frame_data]) # size = 4
                                continue

                            d_angles = extractParams.extract_angle_features(frame_data.skeleton) # single point
                            cur_features_list.append([d_distances, d_angles])
                            dist_visualization, angle_visualization = param_visualization(d_distances, d_angles)

                            if updated_speed:
                                # Find factors impact speed change
                                if not prev_features_list:
                                    prev_features_list = cur_features_list
                                    cur_features_list = []
                                else:
                                    extractParams.extract_pct_change_list(prev_features_list, cur_features_list)

                                    # Save which params have changed the most
                                    pct_dist_changes = get_first_k(extractParams.pct_dist_changes,5)
                                    pct_angle_changes = get_first_k(extractParams.pct_angle_changes,5)
                                    

                                    send_to_client(clientsocket, [second_to_time_str(frame_idx/fps), annotated_frame, frame, 
                                                                  sub_frame_data, dist_visualization, angle_visualization, 
                                                                  pct_dist_changes, pct_angle_changes]) # size = 8

                                    if save_csv:
                                        data =  second_to_time_str(frame_idx/fps) + ',' + str(swimming_speed) + ',' + str(frame_data.speed_pct_change)+ ',' + str(dict_to_string(extractParams.pct_dist_changes, 5)) + ',' + str(dict_to_string(extractParams.pct_angle_changes, 5))
                                        write_to_csv(data, SAVE_CSV_PATH['SOCKET'])

                                    prev_features_list = cur_features_list
                                    cur_features_list = []
                            else:
                                send_to_client(clientsocket, [second_to_time_str(frame_idx/fps), annotated_frame, 
                                                              frame, sub_frame_data, dist_visualization, angle_visualization]) # size = 6

                            if out_video:
                                output.write(annotated_frame)


                        else:
                            if isinstance(DEVICE_ID, str) and os.path.exists(DEVICE_ID):
                                logging.info('Video ended. Replaying...')
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                frame_idx = -1
                                cur_features_list = []
                                prev_features_list = []
                                continue

                            # cap = cv2.VideoCapture(DEVICE_ID)
                            # frame_idx = -1
                            cap.release()
                            output.release()
                            serversocket.close()
                            clientsocket.close()
                            
                            logging.info('Video ended')
                            if save_csv:
                                logging.info(f'Saved csv file to {SAVE_CSV_PATH}')
                                
                            if out_video:
                                logging.debug(f'Annotated video is saved at {out_video}')
                            break

            except Exception as e:
                traceback.print_exc()
                logging.info("Closed a connection from %s" % str(addr))
                clientsocket.close()
                cap.release()
                if out_video:
                    logging.debug(f'Annotated video is saved at {out_video}')


    except KeyboardInterrupt:
        serversocket.close()
        logging.info('Server closed')
        cap.release()
        output.release()
        logging.info(f'Saved video to {out_video}')

def run_socket_multi(debug=False, save_csv=False, out_video=SAVE_VIDEO_PATH['SOCKET'], fx=1.0, fy=1.0):
    # Runtime knobs to reduce CPU overhead
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    cv2.setNumThreads(0)

    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind((SERVER_HOST, SERVER_PORT))
    serversocket.listen(5)

    logging.info("SERVER STARTED (MULTI)")

    try: 
        while True:
            clientsocket, addr = serversocket.accept()
            try:
                if clientsocket:
                    logging.info("Got a connection from %s" % str(addr))

                    # Initialize workers
                    pose_config = read_yaml(POSE_CONFIG)
                    seg_config = read_yaml(SEG_CONFIG)
                    pose_in_q, pose_out_q = queue.Queue(maxsize=4), queue.Queue()
                    seg_in_q,  seg_out_q  = queue.Queue(maxsize=4), queue.Queue()

                    pose_thread = PoseWorker(pose_config['model_path'], pose_in_q, pose_out_q, **pose_config['inference'])
                    seg_thread  = SegWorker(seg_config['model_path'],  seg_in_q,  seg_out_q, **seg_config['inference'])
                    
                    pose_thread.start()
                    seg_thread.start()

                    cap = cv2.VideoCapture(DEVICE_ID)
                    if fx < 1 and fy < 1:
                        frame_width, frame_height = int(cap.get(3)*fx), int(cap.get(4)*fy)
                    else:
                        frame_width, frame_height = fx, fy
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    if fps == 0: fps = 30
                    common_fps = [24, 30, 60, 120]
                    fps = common_fps[np.argmin([abs(i-fps) for i in common_fps])]

                    calculate_frame = MainCalculationMulti(fps=fps) # fps from the original video, normally 60

                    if not os.path.exists(os.path.dirname(out_video)):
                        os.makedirs(os.path.dirname(out_video))
                    output = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'MP4V'),
                                            fps//FPS_RATE, (frame_width, frame_height))

                    frame_idx = -2
                    prev_frame = None
                    frame_keypoints = None
                    lane_divider_bboxes = None

                    while(cap.isOpened()):
                        ret, frame = cap.read()
                        if lane_divider_bboxes is not None:
                            print(f'len lane_divider_bboxes = {len(lane_divider_bboxes)}')
                        if ret:
                            if fx < 1 and fy < 1:
                                frame = cv2.resize(frame, (0, 0), fx=fx, fy=fy)
                            else:
                                frame = cv2.resize(frame, (fx, fy)) 

                            frame_idx += 1
                            if frame_idx % (SAVE_AFTER_SECONDS*fps) == 0:
                                logging.info(f'>>>>> {SAVE_AFTER_SECONDS} seconds elapsed. Current frame idx is {frame_idx}')
                            
                            if frame_idx > 0 and frame_idx % FPS_RATE != 0: 
                                continue

                            init_time = time.perf_counter()
                            # Enqueue to BOTH workers
                            pose_in_q.put((frame, frame_idx))
                            
                            run_seg = (frame_idx < 0) or ((frame_idx + FPS_RATE) % FREQ_SEGMENT == 0) 
                            if run_seg:
                                seg_in_q.put((frame, frame_idx))

                            # Run previous frame's calculation while waiting for current frame's results
                            if frame_idx > -1: 
                                annotated_frame, frame_data, updated_speed, overlap = calculate_frame.swimming_calculation(
                                    frame=prev_frame, 
                                    frame_idx=frame_idx,
                                    frame_keypoints=frame_keypoints, 
                                    lane_divider_bboxes=lane_divider_bboxes,
                                    debug=debug,
                                    from_socket=True)
                                
                            # Deque from BOTH workers
                            frame_idx_p, frame_keypoints, (p0, p1) = pose_out_q.get()
                            
                            if run_seg:
                                frame_idx_s, lane_divider_bboxes,  (s0, s1) = seg_out_q.get()
                            else:
                                s0, s1 = 0, 0

                            if frame_idx > -1:
                                frame_data.pose_time = p1 - p0
                                frame_data.segment_time = s1 - s0
                                
                                torch.cuda.synchronize()
                                frame_data.total_time = time.perf_counter() - init_time

                                if debug:
                                    logging.info(f'Pose time: {(p1 - p0)*1000.0:.2f} ms, Seg time: {(s1 - s0)*1000.0:.2f} ms, Total frame time: {frame_data.total_time*1000.0:.2f} ms')

                                raw_frame_to_client = visualize_swimmer_id(prev_frame, frame_data)
                                send_to_client(clientsocket, [second_to_time_str(frame_idx/fps), annotated_frame, raw_frame_to_client, frame_data])

                                if out_video:
                                    output.write(annotated_frame)

                            process_time = time.perf_counter() - init_time
                            if process_time > (FPS_RATE / fps):
                                frames_to_skip = int(process_time * fps) - 1
                                if frames_to_skip > 0:
                                    logging.warning(f"Processing time {process_time*1000:.1f}ms > {1000.0*FPS_RATE/fps:.1f}ms. Skipping {frames_to_skip} frames.")
                                    for _ in range(frames_to_skip):
                                        cap.grab()
                                        frame_idx += 1

                            prev_frame = frame

                        else:
                            if isinstance(DEVICE_ID, str) and os.path.exists(DEVICE_ID):
                                logging.info('Video ended. Replaying...')
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                frame_idx = -2
                                prev_frame = None
                                frame_keypoints = None
                                lane_divider_bboxes = None
                                continue

                            cap.release()
                            output.release()
                            clientsocket.close()
                            
                            # Stop workers
                            pose_in_q.put(None);  pose_thread.stop_flag.set();  pose_thread.join()
                            seg_in_q.put(None);   seg_thread.stop_flag.set();   seg_thread.join()
                            try:
                                torch.cuda.synchronize()
                            except Exception:
                                pass
                            
                            logging.info('Video ended')
                            if out_video:
                                logging.debug(f'Annotated video is saved at {out_video}')
                            break

            except Exception as e:
                traceback.print_exc()
                logging.info("Closed a connection from %s" % str(addr))
                if 'clientsocket' in locals(): clientsocket.close()
                if 'cap' in locals(): cap.release()
                if 'pose_thread' in locals() and pose_thread.is_alive():
                    pose_in_q.put(None); pose_thread.stop_flag.set(); pose_thread.join()
                if 'seg_thread' in locals() and seg_thread.is_alive():
                    seg_in_q.put(None); seg_thread.stop_flag.set(); seg_thread.join()
                if out_video:
                    logging.debug(f'Annotated video is saved at {out_video}')


    except KeyboardInterrupt:
        serversocket.close()
        logging.info('Server closed')
        if 'cap' in locals(): cap.release()
        if 'output' in locals(): output.release()
        logging.info(f'Saved video to {out_video}')
