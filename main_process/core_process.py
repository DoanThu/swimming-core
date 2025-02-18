from config.general import POSE_CONFIG, FACE_CONFIG, SEG_CONFIG, DETECT_CONFIG, SAVE_AFTER_SECONDS, NO_POINTS_SEGMENTATION, FREQ_SEGMENT
from model_caller.pose_model_caller import PoseCallerYOLO
from model_caller.face_model_caller import FaceCallerYOLO
from model_caller.seg_model_caller import SegCallerYOLO
from model_caller.detection_model_caller import DetectionCallerYOLO
import torch 
import cv2 
import time
from utils.visualize_utils import draw_keypoints, write_texts, draw_segmentation, draw_dot, draw_detection
from utils.file_utils import read_yaml
from utils.skeleton_utils import get_valid_skeletons, get_bbox, get_bbox_area
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
import logging 
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')

class MainCalculation:
    def __init__(self, fps:int, fx:float=1.0, fy:float=1.0, lane_type='segmentation'):
        self.fps = fps
        self.fx = fx 
        self.fy = fy
        self.lane_type = lane_type
        self.frame_data_list: List[FrameData] = []

        self.RANDOM_COLORS = np.random.randint(0, 255, (100, 3))

        self.stroke_process = StrokeProcess()
        self.status_process = StatusProcess()
        self.side_process = SideProcess()
        self.direction_process = DirectionProcess()
        self.speed_process = SpeedProcess(fps=self.fps)
        self.anchor_process = AnchorProcess(window=self.fps*5)
        self.lane_divider_process = LaneDivider()

        self.frame_data_list: List[FrameData] = []

        self.pose_config = read_yaml(POSE_CONFIG)
        self.face_config = read_yaml(FACE_CONFIG)
        self.seg_config = read_yaml(SEG_CONFIG)
        self.detection_config = read_yaml(DETECT_CONFIG)

        self.pose_caller = PoseCallerYOLO(self.pose_config['model_path'])
        self.face_caller = FaceCallerYOLO(self.face_config['model_path'])
        self.seg_caller = SegCallerYOLO(self.seg_config['model_path'])
        self.detect_caller = DetectionCallerYOLO(self.detection_config['model_path'])

        if torch.cuda.is_available():
            logging.info('CUDA is available. Loading pose model from ' + self.pose_config['model_path'])
            logging.info('CUDA is available. Loading face model from ' + self.face_config['model_path'])
            logging.info('CUDA is available. Loading segmentation model from ' + self.seg_config['model_path'])
            logging.info('CUDA is available. Loading detection model from ' + self.detection_config['model_path'])
        else:
            logging.info('CUDA is not available. Using CPU instead.')


        
    def swimming_calculation(self, frame:np.array, frame_idx:int, debug:bool=True) -> tuple:
        updated_speed = False 

        frame = cv2.resize(frame, None, fx=self.fx, fy=self.fy)
        frame_data = FrameData()
    
        if len(self.frame_data_list) != 0:
            frame_data.status = self.frame_data_list[-1].status
            frame_data.frame_orientation = self.frame_data_list[-1].frame_orientation
            frame_data.skeleton = self.frame_data_list[-1].skeleton
            frame_data.red_marker = self.frame_data_list[-1].red_marker
        
        frame_data.frame_idx = frame_idx
        lanes_segmentation = []
        lane_divider_bboxes = []
            
        # call models 
        frame_keypoints = self.pose_caller.get_keypoint(frame, **self.pose_config['inference'])
        # if nothing detected, its shape is (N,0,51) 
        if frame_keypoints.shape[1] != 0:
            frame_keypoints = get_valid_skeletons(frame_keypoints) # filter invalid keypoints
                
            # or (0, 17, 2)
            if frame_keypoints.shape[0] != 0:
                # print(frame_keypoints.shape)
                # get one closest skeleton
                frame_data.skeleton = frame_keypoints
                frame_data.bbox = get_bbox(frame_data.skeleton[0])
                # frame_data.bbox_area = swimmer_tracker.bbox_area.item()
                frame_data.bbox_area = get_bbox_area(frame_data.bbox[0], frame_data.bbox[1], frame_data.bbox[2], frame_data.bbox[3])
                            
                # TODO: get face (can be optimised)
                # face_bboxes = self.face_caller.get_face(frame, **self.face_config['inference'])
                # face_bbox = self.stroke_process.match_face(frame_data.skeleton, face_bboxes)
                # frame_data.face_up = face_bbox != None
                frame_data.face_up = False

                # get full lane divider for frame orientation
                if frame_idx % (10*FREQ_SEGMENT) == 0: # do the following every 10*FREQ_SEGMENT frames
                    if self.lane_type == 'segmentation':
                        lane_divider_bboxes = self.seg_caller.get_lane_dividers(frame, **self.seg_config['inference'])
                    elif self.lane_type == 'detection':
                        lane_divider_bboxes = self.detect_caller.detect_lane_dividers(frame, **self.detection_config['inference'])
                    direction = sum([1 if w > h else -1 for _,_,w,h in lane_divider_bboxes])
                    frame_data.frame_orientation = FrameDataConst.HORIZONTAL if direction >= 0 else FrameDataConst.VERTICAL

                # get skeleton direction
                frame_data.direction = self.direction_process.get_skeleton_direction(frame_data.skeleton)
                
                # detect status only RACE/STOP now
                frame_data.status = self.status_process.get_status(frame_data.skeleton, frame_data, self.frame_data_list,
                                                                   previous_interval=self.fps*3)
                
                # TODO: classify stroke
                # frame_data.stroke = self.stroke_process.classify_stroke(frame_data, self.frame_data_list)
                
                # TODO: fix left right swap
                # frame_data.skeleton = side_process.get_correct_side(frame_data.skeleton, frame_data)
                
                if frame_idx % FREQ_SEGMENT == 0: # do the following every FREQ_SEGMENT frames
                    if self.lane_type == 'segmentation':
                        if len(lane_divider_bboxes) == 0:
                            lane_divider_bboxes = self.seg_caller.get_lane_dividers(frame, **self.seg_config['inference'])
                    elif self.lane_type == 'detection':
                        if len(lane_divider_bboxes) == 0:
                            lane_divider_bboxes = self.detect_caller.detect_lane_dividers(frame, **self.detection_config['inference'])
                    
                    # generate random points inside the lane
                    for x,y,w,h in lane_divider_bboxes:
                        if frame_data.frame_orientation == FrameDataConst.HORIZONTAL:
                            random_x = np.random.randint(0, frame.shape[1], NO_POINTS_SEGMENTATION)
                            random_y = np.random.randint(y, y+h, NO_POINTS_SEGMENTATION)
                        else:
                            random_x = np.random.randint(x, x+w, NO_POINTS_SEGMENTATION)
                            random_y = np.random.randint(0, frame.shape[0], NO_POINTS_SEGMENTATION)
                        self.anchor_process.add_random_anchor_points(frame_idx, random_x, random_y)

                    self.anchor_process.update_anchor_points(frame_idx, frame, frame_data, lane_divider_bboxes, divider_type=self.lane_type)


                    # calculate speed
                    self.speed_process.calculate_speed(frame, frame_data, self.anchor_process.anchor_list,
                                                       lane_divider_bboxes, lane_type=self.lane_type, unit_size=1) # unit_size=1, counting pixel
                    frame_data.speed = self.speed_process.current_speed
                    frame_data.speed_pct_change = self.speed_process.pct_change
                    frame_data.red_marker = self.speed_process.red_marker
                    updated_speed = True

                else:
                    self.anchor_process.update_anchor_points(frame_idx, frame, frame_data, [])
                    frame_data.speed = self.frame_data_list[-1].speed
                    frame_data.speed_pct_change = self.frame_data_list[-1].speed_pct_change


                        
            
        if torch.is_tensor(frame_data.skeleton):
            frame_data.skeleton = frame_data.skeleton.cpu().numpy().tolist()
        frame_data.bbox = [_coord.item() for _coord in frame_data.bbox]
        # append current frame to list
        self.frame_data_list.append(frame_data)

        annotated_frame = frame.copy()
        annotated_frame = draw_keypoints(frame, frame_data.skeleton)
        # if self.lane_type == 'segmentation':
            # annotated_frame = draw_segmentation(annotated_frame, lane_divider_bboxes)
        # elif self.lane_type == 'detection':
            # annotated_frame = draw_detection(annotated_frame, lane_divider_bboxes)

        if debug:
            texts = frame_data.__str__()
            annotated_frame = write_texts(frame, texts, 30, org=(30,30))

        
        for k,v in self.anchor_process.anchor_list.items():
            annotated_frame = draw_dot(annotated_frame, v, 
                                        color=self.RANDOM_COLORS[k%len(self.RANDOM_COLORS)].tolist(), radius=4)
            
        if len(self.frame_data_list) > self.fps * SAVE_AFTER_SECONDS:
            self.frame_data_list.pop(0)
            # self.frame_data_list = self.frame_data_list[-SAVE_AFTER_SECONDS*self.fps:]
            
        return annotated_frame, frame_data, updated_speed
    
