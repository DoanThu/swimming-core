from config.general import POSE_CONFIG, SEG_CONFIG, SAVE_AFTER_SECONDS, NO_POINTS_SEGMENTATION, FREQ_SEGMENT, PIXEL_DENSITY_WIDTH, PIXEL_DENSITY_HEIGHT, TIME_WINDOW_STROKE
from model_caller.pose_model_caller import PoseCallerYOLO
from model_caller.face_model_caller import FaceCallerYOLO
from model_caller.seg_model_caller import SegCallerYOLO
from model_caller.detection_model_caller import DetectionCallerYOLO
import torch 
import cv2 
from utils.lane_segment_utils import get_ground, bbox_overlap_or_near
from utils.visualize_utils import draw_keypoints, write_texts, draw_segmentation, draw_dot, draw_detection
from utils.file_utils import read_yaml
from utils.skeleton_utils import get_valid_skeletons, get_bbox, get_bbox_area, get_mid_skeleton
from postprocess.single_data import FrameData, FrameDataConst
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
    def convert_pixel_to_meter(self, bboxes:np.array, frame_orientation:int, skeletons: torch.Tensor) -> None:
        def is_valid_consecutive_lanes(lane1, lane2, skeletons, axis):
            count = 0
            for skeleton in skeletons:
                if lane1 <= skeleton[0][axis] <= lane2:
                    count += 1
            return count == 1
        # if len(self.pixel_to_meters) > 0: return

        distances = []

        if frame_orientation == FrameDataConst.HORIZONTAL:
            bboxes_sorted = sorted(bboxes, key=lambda box: box[1])

            for i in range(len(bboxes_sorted) - 1):
                y1, h1 = bboxes_sorted[i][1], bboxes_sorted[i][3]
                y2 = bboxes_sorted[i + 1][1]
                distance = y2 - y1
                # if not is_valid_consecutive_lanes(y1, y2, skeletons, 1): continue # make sure 1 lane has 1 skeleton
                if distance < 30: continue
                distances.append(distance)
        elif frame_orientation == FrameDataConst.VERTICAL:
            bboxes_sorted = sorted(bboxes, key=lambda box: box[0])

            for i in range(len(bboxes_sorted) - 1):
                x1, w1 = bboxes_sorted[i][0], bboxes_sorted[i][2]
                x2 = bboxes_sorted[i + 1][0]
                # if not is_valid_consecutive_lanes(x1, x2, skeletons, 0): continue  # make sure 1 lane has 1 skeleton
                distance = x2 - x1
                if distance < 30: continue
                distances.append(distance)
        temp = np.mean(distances)/2.5*PIXEL_DENSITY_HEIGHT/PIXEL_DENSITY_WIDTH
        if temp == 0: return

        if len(self.pixel_to_meters) == 0:
            self.pixel_to_meters.append(temp)
        else:
            if abs(temp - np.mean(self.pixel_to_meters))/temp < 0.1:
                self.pixel_to_meters.append(temp)
        if len(self.pixel_to_meters) > 10:
            self.pixel_to_meters.pop(0)

    def __init__(self, fps:int):
        self.pixel_to_meters = []

        self.fps = fps
        self.frame_data_list: List[FrameData] = []

        self.RANDOM_COLORS = np.random.randint(0, 255, (100, 3))

        self.frame_data_list: List[FrameData] = []

        self.pose_config = read_yaml(POSE_CONFIG)
        self.seg_config = read_yaml(SEG_CONFIG)
        # self.face_config = read_yaml(FACE_CONFIG)
        # self.detection_config = read_yaml(DETECT_CONFIG)


        self.pose_caller = PoseCallerYOLO(self.pose_config['model_path'])
        self.seg_caller = SegCallerYOLO(self.seg_config['model_path'])

        # self.face_caller = FaceCallerYOLO(self.face_config['model_path'])
        # self.detect_caller = DetectionCallerYOLO(self.detection_config['model_path'])


        if torch.cuda.is_available():
            logging.info('CUDA is available. Loading pose model from ' + self.pose_config['model_path'])
            logging.info('CUDA is available. Loading segmentation model from ' + self.seg_config['model_path'])
            # logging.info('CUDA is available. Loading face model from ' + self.face_config['model_path'])
            # logging.info('CUDA is available. Loading detection model from ' + self.detection_config['model_path'])

        else:
            logging.info('CUDA is not available. Using CPU instead.')


        self.stroke_process = StrokeProcess(fps=self.fps)
        self.status_process = StatusProcess()
        self.side_process = SideProcess()
        self.direction_process = DirectionProcess()
        self.speed_process = SpeedProcess(fps=self.fps)
        self.anchor_process = AnchorProcess(window=self.fps*5)
        self.lane_divider_process = LaneDivider()


        self.prev_frame = np.array([])


        
    def swimming_calculation(self, frame:np.array, frame_idx:int, debug:bool=True) -> tuple:
        if len(self.prev_frame) == 0:
            self.prev_frame = frame

        updated_speed = False 
        overlap = False

        frame_data = FrameData()
    
        if len(self.frame_data_list) != 0:
            frame_data.status = self.frame_data_list[-1].status
            frame_data.frame_orientation = self.frame_data_list[-1].frame_orientation
            frame_data.skeleton = self.frame_data_list[-1].skeleton
            frame_data.red_marker = self.frame_data_list[-1].red_marker
        
        frame_data.frame_idx = frame_idx
        lane_divider_bboxes = []
        bbox_ground = []

         # get full lane divider for frame orientation
        if frame_idx % (10*FREQ_SEGMENT) == 0: # do the following every 10*FREQ_SEGMENT frames
            lane_divider_bboxes = self.seg_caller.get_lane_dividers(frame, **self.seg_config['inference'])
            direction = sum([1 if w > h else -1 for _,_,w,h in lane_divider_bboxes])
            frame_data.frame_orientation = FrameDataConst.HORIZONTAL if direction >= 0 else FrameDataConst.VERTICAL
            
        # call models 
        frame_keypoints = self.pose_caller.get_keypoints(frame, **self.pose_config['inference'])
        # if nothing detected, its shape is (N,0,51), the valid skeletons return (0, 17,32)
        if frame_keypoints.shape[1] != 0:
            # and get_valid_skeletons(frame_keypoints).shape[0] != 0:
            # get one closest skeleton

            # if frame_data.frame_orientation == FrameDataConst.HORIZONTAL:
                # reference_line = frame.shape[0] // 2
            # elif frame_data.frame_orientation == FrameDataConst.VERTICAL:
                # reference_line = frame.shape[1] // 2

            selected_skeleton = get_mid_skeleton(frame_keypoints, frame_data.frame_orientation, width=frame.shape[1], height=frame.shape[0])
        
        if  frame_keypoints.shape[1] != 0 and selected_skeleton.shape[0] != 0: 
            frame_data.skeleton = selected_skeleton
            frame_data.bbox = get_bbox(frame_data.skeleton[0]) # x y w h
            # frame_data.bbox_area = get_bbox_area(frame_data.bbox[0], frame_data.bbox[1], frame_data.bbox[2], frame_data.bbox[3])
            frame_data.bbox_area = frame_data.bbox[2] * frame_data.bbox[3]

            if frame_idx % (10*FREQ_SEGMENT) == 0: # do the following every 10*FREQ_SEGMENT frames
                self.convert_pixel_to_meter(lane_divider_bboxes, frame_data.frame_orientation, frame_keypoints)

            bbox_ground = get_ground(frame) # x y w h
            if bbox_ground[0] != -1 and frame_data.bbox[0] != -1:
                overlap = bbox_overlap_or_near(frame_data.bbox, bbox_ground, threshold=20)
            else:
                overlap = False

            # TODO: get face (can be optimised)
            # face_bboxes = self.face_caller.get_face(frame, **self.face_config['inference'])
            # face_bbox = self.stroke_process.match_face(frame_data.skeleton, face_bboxes)
            # frame_data.face_up = face_bbox != None
            frame_data.face_up = False
           

            # get skeleton direction
            frame_data.direction = self.direction_process.get_skeleton_direction(frame_data.skeleton)
            
            # detect status only RACE/STOP now
            # frame_data.status = self.status_process.get_status(frame_data.skeleton, frame_data, self.frame_data_list,
                                                                # previous_interval=self.fps*3)
            
            # count stroke
            if len(self.frame_data_list) < TIME_WINDOW_STROKE:
                frame_data.stroke_count = 0
            else:
                temp_frame_data_list = []
                for i in range(len(self.frame_data_list)-1, len(self.frame_data_list)-TIME_WINDOW_STROKE-1, -1):
                    temp_frame_data_list.append(self.frame_data_list[i].skeleton[0])
                temp_frame_data_list = np.array(temp_frame_data_list)
                # print(temp_frame_data_list.shape)
                frame_data.stroke_count = self.stroke_process.count_stroke(temp_frame_data_list)
            
            # TODO: classify stroke
            # frame_data.stroke = self.stroke_process.classify_stroke(frame_data, self.frame_data_list)
            
            # TODO: fix left right swap
            # frame_data.skeleton = side_process.get_correct_side(frame_data.skeleton, frame_data)
                
            if frame_idx % FREQ_SEGMENT == 0: # do the following every FREQ_SEGMENT frames
                if len(lane_divider_bboxes) == 0:
                    lane_divider_bboxes = self.seg_caller.get_lane_dividers(frame, **self.seg_config['inference'])
                
                # generate random points inside the lane
                for x,y,w,h in lane_divider_bboxes:
                    if frame_data.frame_orientation == FrameDataConst.HORIZONTAL:
                        random_x = np.random.randint(0, frame.shape[1], NO_POINTS_SEGMENTATION)
                        random_y = np.random.randint(y, y+h, NO_POINTS_SEGMENTATION)
                    else:
                        random_x = np.random.randint(x, x+w, NO_POINTS_SEGMENTATION)
                        random_y = np.random.randint(0, frame.shape[0], NO_POINTS_SEGMENTATION)
                    self.anchor_process.add_random_anchor_points(frame_idx, random_x, random_y)

                self.anchor_process.update_anchor_points(frame_idx, frame, self.prev_frame, frame_data, lane_divider_bboxes)

                # calculate speed
                self.speed_process.calculate_speed(frame, frame_data, self.anchor_process.anchor_list,
                                                    lane_divider_bboxes) # return speed in pixels
                frame_data.speed_px = self.speed_process.current_speed
                frame_data.speed_m = frame_data.speed_px/np.mean(self.pixel_to_meters) # convert to meters
                frame_data.speed_pct_change = self.speed_process.pct_change
                frame_data.red_marker = self.speed_process.red_marker
                updated_speed = True

                distance_swum = frame_data.speed_m 
                stroke_count = frame_data.stroke_count
                distance_per_stroke = distance_swum/stroke_count * 60 if stroke_count != 0 else 0
                frame_data.distance_per_stroke = round(distance_per_stroke,2)

            else:
                self.anchor_process.update_anchor_points(frame_idx, frame, self.prev_frame, frame_data, [])
                frame_data.speed_m = self.frame_data_list[-1].speed_m
                frame_data.speed_px = self.frame_data_list[-1].speed_px
                frame_data.speed_pct_change = self.frame_data_list[-1].speed_pct_change
                frame_data.distance_per_stroke = self.frame_data_list[-1].distance_per_stroke


        else:
            self.anchor_process.update_anchor_points(frame_idx, frame, self.prev_frame, frame_data, [])
            if self.frame_data_list:
                frame_data.speed_m = self.frame_data_list[-1].speed_m
                frame_data.speed_px = self.frame_data_list[-1].speed_px
                frame_data.speed_pct_change = self.frame_data_list[-1].speed_pct_change
                frame_data.distance_per_stroke = self.frame_data_list[-1].distance_per_stroke

        self.prev_frame = frame
        
                        
            
        if torch.is_tensor(frame_data.skeleton):
            frame_data.skeleton = frame_data.skeleton.cpu().numpy().tolist()
        # append current frame to list
        self.frame_data_list.append(frame_data)

        annotated_frame = frame.copy()
        annotated_frame = draw_keypoints(annotated_frame, frame_data.skeleton, thickness=1)

        if debug:
            annotated_frame = draw_detection(annotated_frame, lane_divider_bboxes)
            if len(bbox_ground) != 0 and bbox_ground[0] != -1:
                annotated_frame = draw_detection(annotated_frame, [bbox_ground])
            texts = frame_data.__str__()
            annotated_frame = write_texts(annotated_frame, texts, 30, org=(30,30))

        
        for k,v in self.anchor_process.anchor_list.items():
            if debug:
                frame_idx_ = str(k)
            else:
                frame_idx_ = ''
            annotated_frame = draw_dot(annotated_frame, v, 
                                        color=self.RANDOM_COLORS[k%len(self.RANDOM_COLORS)].tolist(), radius=2,
                                        frame_idx=frame_idx_)
            
        if len(self.frame_data_list) > self.fps * SAVE_AFTER_SECONDS:
            self.frame_data_list.pop(0)
            
        return annotated_frame, frame_data, updated_speed, overlap
    
