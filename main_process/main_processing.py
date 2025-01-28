from config.general import POSE_CONFIG, FACE_CONFIG, SEG_CONFIG, SAVE_AFTER_SECONDS
from model_caller.pose_model_caller import PoseCallerYOLO
from model_caller.face_model_caller import FaceCallerYOLO
from model_caller.seg_model_caller import SegCallerYOLO
import torch 
import cv2 
from utils.visualize_utils import draw_keypoints, write_texts, draw_segmentation, draw_dot
from utils.file_utils import read_yaml
from utils.skeleton_utils import get_valid_skeletons, get_bbox, get_bbox_area
from postprocess.data import FrameData
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

class MainCalculation:
    def __init__(self, fps:int, fx:float=1.0, fy:float=1.0):
        if torch.cuda.is_available():
            logging.info('CUDA is available. Loading pose model from ' + pose_config['model_path'])
            logging.info('CUDA is available. Loading face model from ' + face_config['model_path'])
            logging.info('CUDA is available. Loading segmentation model from ' + seg_config['model_path'])
        else:
            logging.info('CUDA is not available. Using CPU instead.')

        self.fps = fps
        self.fx = fx 
        self.fy = fy
        self.frame_data_list: List[FrameData] = []

        
    def swimming_calculation(self, frame:np.array, frame_idx:int, debug:bool=True) -> tuple:
        frame = cv2.resize(frame, None, fx=self.fx, fy=self.fy)
        frame_data = FrameData()
    
        if len(self.frame_data_list) != 0:
            frame_data.status = self.frame_data_list[-1].status
            frame_data.frame_orientation = self.frame_data_list[-1].frame_orientation
            frame_data.skeleton = self.frame_data_list[-1].skeleton
            frame_data.red_marker = self.frame_data_list[-1].red_marker
        
        frame_data.frame_idx = frame_idx
        lanes_segmentation = []
            
        # call models 
        frame_keypoints = pose_caller.get_keypoint(frame, **pose_config['inference'])
        # if nothing detected, its shape is (N,0,51) 
        if frame_keypoints.shape[1] != 0:
            # print(frame_keypoints) 
            frame_keypoints = get_valid_skeletons(frame_keypoints) # filter invalid keypoints
            # print(frame_keypoints.shape) 
                
            # or (0, 17, 2)
            if frame_keypoints.shape[0] != 0:
                # print(frame_keypoints.shape)
                # get one closest skeleton
                frame_data.skeleton = frame_keypoints
                frame_data.bbox = get_bbox(frame_data.skeleton[0])
                # frame_data.bbox_area = swimmer_tracker.bbox_area.item()
                frame_data.bbox_area = get_bbox_area(frame_data.bbox[0], frame_data.bbox[1], frame_data.bbox[2], frame_data.bbox[3])
                            
                # TODO: get face (can be optimised)
                # face_bboxes = face_caller.get_face(frame, **face_config['inference'])
                # face_bbox = stroke_process.match_face(frame_data.skeleton, face_bboxes)
                # frame_data.face_up = face_bbox != None
                frame_data.face_up = False

                # get full lane divider for frame direction
                if frame_idx % (2*(int(self.fps*seg_config['freq']))) == 0: # do the following every seg_config['freq']*2 second(s)
                    lane_divider_process.set_lane_divider_info(frame, seg_caller, **seg_config['inference'])
                    frame_data.frame_orientation = lane_divider_process.orientation

                # get skeleton direction
                # print(frame_data.skeleton)
                frame_data.direction = direction_process.get_skeleton_direction(frame_data.skeleton)
                
                # detect status
                frame_data.status = status_process.get_status(frame_data.skeleton, frame_data, self.frame_data_list,
                                                            previous_interval=self.fps*3)
                
                # TODO: classify stroke
                # frame_data.stroke = stroke_process.classify_stroke(frame_data, frame_data_list)
                
                # TODO: fix left right swap
                # frame_data.skeleton = side_process.get_correct_side(frame_data.skeleton, frame_data)
                
                # get lane segmentation based on the head
                if frame_idx % (int(self.fps*seg_config['freq'])) == 0: # do the following every seg_config['freq'] second(s)
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
                    frame_data.speed = self.frame_data_list[-1].speed
                    frame_data.pct_change = self.frame_data_list[-1].pct_change
                        
            
        if torch.is_tensor(frame_data.skeleton):
            frame_data.skeleton = frame_data.skeleton.cpu().numpy().tolist()
        frame_data.bbox = [_coord.item() for _coord in frame_data.bbox]
        # append current frame to list
        self.frame_data_list.append(frame_data)
        

        annotated_frame = draw_keypoints(frame, frame_data.skeleton)
        annotated_frame = draw_segmentation(annotated_frame, lanes_segmentation)

        if debug:
            texts = frame_data.__str__()
            annotated_frame = write_texts(frame, texts, 30, org=(30,30))


        
        for k,v in anchor_process.anchor_list.items():
            annotated_frame = draw_dot(annotated_frame, v, 
                                        color=RANDOM_COLORS[k%len(RANDOM_COLORS)].tolist(), radius=4)
            
        if len(self.frame_data_list) > self.fps * SAVE_AFTER_SECONDS:
            self.frame_data_list = self.frame_data_list[-SAVE_AFTER_SECONDS*self.fps:]
            
        return annotated_frame, frame_data


    
    
   