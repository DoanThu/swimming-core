import socket,cv2,pickle, struct
from config.general import FPS_RATE, DEVICE_ID, SAVE_AFTER_SECONDS, SAVE_VIDEO_PATH, SAVE_CSV_PATH, REAL_SIZE_WIDTH
import logging 
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')
from main_process.core_process import MainCalculation
from postprocess.analytics import ExtractParams
import os
from utils.dict_utils import dict_to_string, get_first_k
import traceback
from utils.time_utils import second_to_time_str
import numpy as np
from utils.file_utils import write_to_csv
from postprocess.data import FrameDataConst


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

def run_socket(debug=False, save_json=False, save_csv=False, out_video=SAVE_VIDEO_PATH, fx=1.0, fy=1.0):
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = socket.gethostname()
    port = 9999
    serversocket.bind((host, port))
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

                            annotated_frame, frame_data, updated_speed, overlap = calculate_frame.swimming_calculation(frame=frame, frame_idx=frame_idx, debug=debug)
                            
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
                                        write_to_csv(data, SAVE_CSV_PATH)

                                    prev_features_list = cur_features_list
                                    cur_features_list = []
                            else:
                                send_to_client(clientsocket, [second_to_time_str(frame_idx/fps), annotated_frame, 
                                                              frame, sub_frame_data, dist_visualization, angle_visualization]) # size = 6

                            # if out_video:
                                # output.write(annotated_frame)


                        else:
                            cap = cv2.VideoCapture(DEVICE_ID)
                            frame_idx = -1
                            # cap.release()
                            # output.release()
                            # serversocket.close()
                            # clientsocket.close()
                            
                            # logging.info('Video ended')
                            # if save_csv:
                            #     logging.info(f'Saved csv file to {SAVE_CSV_PATH}')
                                
                            # if out_video:
                            #     logging.debug(f'Annotated video is saved at {out_video}')
                            # break

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
