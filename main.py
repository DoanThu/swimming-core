from config.general import POSE_CONFIG, FACE_CONFIG, SEG_CONFIG, SAVE_AFTER_FRAMES, SAVE_JSON_PATH, SAVE_CSV_PATH, SWIMMER_POSITION_X, SWIMMER_POSITION_Y, VIDEO_PATH
from postprocess.tracking import SwimmerTracking
from model_caller.pose_model_caller import PoseCallerYOLO
from model_caller.face_model_caller import FaceCallerYOLO
from model_caller.seg_model_caller import SegCallerYOLO
import torch 
import cv2 
from utils.visualize_utils import draw_keypoints, write_texts, draw_segmentation, draw_dot
from utils.file_utils import read_yaml, write_to_csv
from utils.skeleton_utils import get_valid_skeletons
from postprocess.data import FrameData, FrameDataConst
from postprocess.stroke_process import StrokeProcess
from postprocess.status_process import StatusProcess
from postprocess.side_process import SideProcess
from postprocess.direction_process import DirectionProcess
from postprocess.speed_process import SpeedProcess
from postprocess.anchor_process import AnchorProcess
from postprocess.lane_divider_process import LaneDivider
from typing import List
import numpy as np
import os
import logging 
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')

frame_data_list: List[FrameData] = []

pose_config = read_yaml(POSE_CONFIG)
face_config = read_yaml(FACE_CONFIG)
seg_config = read_yaml(SEG_CONFIG)

pose_caller = PoseCallerYOLO(pose_config['model_path'])
face_caller = FaceCallerYOLO(face_config['model_path'])
seg_caller = SegCallerYOLO(seg_config['model_path'])

stroke_process = StrokeProcess()
status_process = StatusProcess()
side_process = SideProcess()
direction_process = DirectionProcess()
speed_process = SpeedProcess()
anchor_process = AnchorProcess(window=120)
lane_divider_process = LaneDivider()

RANDOM_COLORS = np.random.randint(0, 255, (100, 3))


def run(debug=False, save_json=False, save_csv=False, out_video='output.mp4', fx=1, fy=1):
    if torch.cuda.is_available():
        logging.info('CUDA is available. Loading pose model from ' + pose_config['model_path'])
        logging.info('CUDA is available. Loading face model from ' + face_config['model_path'])
        logging.info('CUDA is available. Loading segmentation model from ' + seg_config['model_path'])
    else:
        logging.info('CUDA is not available. Using CPU instead.')
    
    # swimmer_position = pose_config['swimmer_position']
    swimmer_position = {}
    swimmer_position['x'] = SWIMMER_POSITION_X*fx
    swimmer_position['y'] = SWIMMER_POSITION_Y*fy
    swimmer_tracker = SwimmerTracking(swimmer_position)
    
    # cap = cv2.VideoCapture(pose_config['video_path'])
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_width, frame_height = int(cap.get(3)*fx), int(cap.get(4)*fy)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_idx = -1
    output = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'MP4V'),
                             fps, (frame_width, frame_height))
    
    if save_csv:
        if os.path.exists(SAVE_CSV_PATH):
            os.remove(SAVE_CSV_PATH)
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if ret:
            frame = cv2.resize(frame, None, fx=fx, fy=fy)
            frame_idx += 1
            frame_data = FrameData()
            
            if len(frame_data_list) != 0:
                frame_data.status = frame_data_list[-1].status
                frame_data.frame_orientation = frame_data_list[-1].frame_orientation
                frame_data.skeleton = frame_data_list[-1].skeleton
                frame_data.red_marker = frame_data_list[-1].red_marker
                
            frame_data.frame_idx = frame_idx
            lanes_segmentation = []
            
            # call models 
            frame_keypoints = pose_caller.get_keypoint(frame, **pose_config['inference'])
            # print(frame_keypoints.shape)
            # if nothing detected, its shape is (N,0,51) 
            if frame_keypoints.shape[1] != 0:
                # print(frame_keypoints) 
                frame_keypoints = get_valid_skeletons(frame_keypoints) # filter invalid keypoints
                # print(frame_keypoints.shape) 
                
                # or (0, 17, 2)
                if frame_keypoints.shape[0] != 0:
                    # print(frame_keypoints.shape)
                    # get one closest skeleton
                    swimmer_tracker.get_closest_skeleton(frame_keypoints)
                    swimmer_skeleton = swimmer_tracker.skeleton
                    if swimmer_skeleton.shape[0] != 0: # if the skeleton at the tracked point is found
                        frame_data.skeleton = swimmer_skeleton
                        frame_data.bbox = swimmer_tracker.bbox
                        frame_data.bbox_area = swimmer_tracker.bbox_area.item()
                        
                        # TODO: get face (can be optimised)
                        # face_bboxes = face_caller.get_face(frame, **face_config['inference'])
                        # face_bbox = stroke_process.match_face(frame_data.skeleton, face_bboxes)
                        # frame_data.face_up = face_bbox != None
                        frame_data.face_up = False

                        # get full lane divider for frame direction
                        if frame_idx % (2*(int(fps*seg_config['freq']))) == 0: # do the following every seg_config['freq']*2 second(s)
                            lane_divider_process.set_lane_divider_info(frame, seg_caller, **seg_config['inference'])
                            frame_data.frame_orientation = lane_divider_process.orientation

                        # get skeleton direction
                        frame_data.direction = direction_process.get_skeleton_direction(frame_data.skeleton)
                        
                        # detect status
                        frame_data.status = status_process.get_status(frame_data.skeleton, frame_data, frame_data_list,
                                                                    previous_interval=fps*3)
                        
                        # TODO: classify stroke
                        # frame_data.stroke = stroke_process.classify_stroke(frame_data, frame_data_list)
                        
                        # TODO: fix left right swap
                        # frame_data.skeleton = side_process.get_correct_side(frame_data.skeleton, frame_data)
                        
                        # get lane segmentation based on the head
                        if frame_idx % (int(fps*seg_config['freq'])) == 0: # do the following every seg_config['freq'] second(s)
                            lanes_segmentation = anchor_process.segment_lane_dividers(frame, frame_data, 
                                                                                    seg_caller, **seg_config['inference'])
                            anchor_process.update_anchor_points(frame_idx, frame, frame_data, lanes_segmentation)
                            # calculate speed
                            speed_process.calculate_speed(frame, frame_data, anchor_process.anchor_list, lanes_segmentation, unit_size=1) # unit_size=1, counting pixel
                            frame_data.speed = speed_process.current_speed
                            frame_data.pct_change = speed_process.pct_change
                            frame_data.red_marker = speed_process.red_marker
                        else:
                            anchor_process.update_anchor_points(frame_idx, frame, frame_data, lanes_segmentation)
                            frame_data.speed = frame_data_list[-1].speed
                            frame_data.pct_change = frame_data_list[-1].pct_change
                    
            
            if torch.is_tensor(frame_data.skeleton):
                frame_data.skeleton = frame_data.skeleton.cpu().numpy().tolist()
            frame_data.bbox = [_coord.item() for _coord in frame_data.bbox]
            # append current frame to list
            frame_data_list.append(frame_data)
                    
            # save SAVE_AFTER_FRAMES frames in json format
            if frame_idx % SAVE_AFTER_FRAMES == 0:
                if save_json:
                    filename = SAVE_JSON_PATH.format(frame_idx//SAVE_AFTER_FRAMES)
                    save_json([_frame.__dict__ for _frame in frame_data_list[frame_idx-SAVE_AFTER_FRAMES:]], filename)
                    if debug:
                        logging.info(f'Saved json file to {filename}')
                else:
                    if debug:
                        logging.debug('>>>>> {} seconds elapsed'.format(SAVE_AFTER_FRAMES/fps))
                    
            if save_csv:
                data = str(frame_data.red_marker) + ',' + str(frame_data.direction) + ',' + str(frame_data.speed) + ',' + ','.join(str(p) for p in frame_data.skeleton)
                write_to_csv(data, SAVE_CSV_PATH)
            
            # For debugging
            if debug:
                annotated_frame = draw_keypoints(frame, frame_data.skeleton)
                texts = frame_data.__str__()
                annotated_frame = write_texts(frame, texts, 30, org=(30,30))
                annotated_frame = draw_segmentation(annotated_frame, lanes_segmentation)
                
                for k,v in anchor_process.anchor_list.items():
                    annotated_frame = draw_dot(annotated_frame, v, 
                                               color=RANDOM_COLORS[k%len(RANDOM_COLORS)].tolist(), radius=4)
                
                output.write(annotated_frame)
            
                # if frame_idx > 20*fps:
                    # break
        else:
            break
        
    cap.release()
    output.release()
    
    if debug:
        logging.debug(f'Annotated video is saved at {out_video}')
    if save_csv:
        logging.debug(f'Csv file is saved at {SAVE_CSV_PATH}')


if __name__ == '__main__':
    """ run the input video frame by frame and save the frame info as json files
    """
    run(debug=True, save_json=False, save_csv=True)
    
    
   