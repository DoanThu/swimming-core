import socket,cv2,pickle, struct
from config.general import FPS_RATE, DEVICE_ID, SAVE_AFTER_SECONDS, SAVE_VIDEO_PATH, SAVE_CSV_PATH
import logging 
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')
from main_process.core_process import MainCalculation
from postprocess.analytics import ExtractParams
import os
from utils.dict_utils import get_first_k
import traceback
from utils.time_utils import second_to_time_str
import numpy as np


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

def run_socket(debug=False, save_json=False, save_csv=False, out_video=SAVE_VIDEO_PATH, fx=1, fy=1, lane_type='segmentation'):
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
                    frame_width, frame_height = int(cap.get(3)*fx), int(cap.get(4)*fy)
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    common_fps = [24, 30, 60, 120]
                    fps = common_fps[np.argmin([abs(i-fps) for i in common_fps])]

                    calculate_frame = MainCalculation(fps=fps//FPS_RATE, lane_type=lane_type)

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
                            frame = cv2.resize(frame, (0, 0), fx=fx, fy=fy)

                            if frame_idx % (SAVE_AFTER_SECONDS*fps) == 0:
                                logging.info(f'>>>>> {SAVE_AFTER_SECONDS} seconds elapsed. Current frame idx is {frame_idx}')
                            
                            if frame_idx % FPS_RATE != 0: 
                                continue

                            annotated_frame, frame_data, updated_speed = calculate_frame.swimming_calculation(frame=frame, frame_idx=frame_idx, debug=debug)
                            
                            sub_frame_data = {'speed': frame_data.speed, 'pct_change': frame_data.speed_pct_change}

                            if not frame_data.skeleton: 
                                send_to_client(clientsocket, [second_to_time_str(frame_idx/fps), annotated_frame, frame, sub_frame_data]) # size = 4
                                continue

                            d_distances = extractParams.extract_distance_features(frame_data.skeleton) # single point
                            d_angles = extractParams.extract_angle_features(frame_data.skeleton) # single point
                            cur_features_list.append([d_distances, d_angles])
                            dist_visualization, angle_visualization = param_visualization(d_distances, d_angles)

                            send_to_client(clientsocket, [second_to_time_str(frame_idx/fps), annotated_frame, 
                                                          frame, sub_frame_data, dist_visualization, angle_visualization]) # size = 6

                            if updated_speed:
                                pass

                            if out_video:
                                output.write(annotated_frame)


                        else:
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
                output.release()
                if out_video:
                    logging.debug(f'Annotated video is saved at {out_video}')


    except KeyboardInterrupt:
        serversocket.close()
        logging.info('Server closed')
        cap.release()
        output.release()
        logging.info(f'Saved video to {out_video}')
