from config.general import POSE_CONFIG, FACE_CONFIG, SEG_CONFIG, DETECT_CONFIG, SAVE_AFTER_SECONDS, NO_POINTS_SEGMENTATION, FREQ_SEGMENT, PIXEL_DENSITY_WIDTH, PIXEL_DENSITY_HEIGHT, TIME_WINDOW_STROKE
from model_caller.pose_model_caller import PoseCallerYOLO
from model_caller.face_model_caller import FaceCallerYOLO
from model_caller.seg_model_caller import SegCallerYOLO
from model_caller.detection_model_caller import DetectionCallerYOLO
import torch 
import cv2 
from utils.lane_segment_utils import get_ground, bbox_overlap_or_near
from utils.visualize_utils import draw_keypoints, write_texts, draw_segmentation, draw_dot, draw_detection
from utils.file_utils import read_yaml
from utils.skeleton_utils import get_valid_skeletons_from_track, get_bbox
from postprocess.multiple_data import FrameMultipleData
from postprocess.const import FrameDataConst
from postprocess.stroke_process import StrokeProcess
from postprocess.status_process import StatusProcess
from postprocess.side_process import SideProcess
from postprocess.direction_process import DirectionProcess
from postprocess.speed_process_v2 import SpeedProcess
from postprocess.anchor_process_v2 import AnchorProcess
from postprocess.lane_divider_process import LaneDivider
from typing import List
import torch
import numpy as np
import logging 
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')

class MainCalculation:
    def convert_pixel_to_meter(self, bboxes:np.array, frame_orientation:int, skeletons: torch.Tensor) -> None:
        distances = []

        if frame_orientation == FrameDataConst.HORIZONTAL:
            bboxes_sorted = sorted(bboxes, key=lambda box: box[1])

            for i in range(len(bboxes_sorted) - 1):
                y1, h1 = bboxes_sorted[i][1], bboxes_sorted[i][3]
                y2 = bboxes_sorted[i + 1][1]
                distance = y2 - y1
                if distance < 30: continue
                distances.append(distance)
        elif frame_orientation == FrameDataConst.VERTICAL:
            bboxes_sorted = sorted(bboxes, key=lambda box: box[0])

            for i in range(len(bboxes_sorted) - 1):
                x1, w1 = bboxes_sorted[i][0], bboxes_sorted[i][2]
                x2 = bboxes_sorted[i + 1][0]
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
        self.frame_data_list = [] # list of FrameMultipleData

        self.RANDOM_COLORS = np.random.randint(0, 255, (100, 3))

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

        frame_data = FrameMultipleData()
    
        if len(self.frame_data_list) != 0:
            frame_data.status_list = self.frame_data_list[-1].status_list.copy()
            frame_data.frame_orientation = self.frame_data_list[-1].frame_orientation
            frame_data.skeleton_list = self.frame_data_list[-1].skeleton_list.copy()
            # frame_data.red_marker_list = self.frame_data_list[-1].red_marker_list.copy()
        
        frame_data.frame_idx = frame_idx
        lane_divider_bboxes = []
        bbox_ground = []

        # get full lane divider for frame orientation
        if frame_idx % (10*FREQ_SEGMENT) == 0: # do the following every 10*FREQ_SEGMENT frames
            lane_divider_bboxes = self.seg_caller.get_lane_dividers(frame, **self.seg_config['inference'])
            direction = sum([1 if w > h else -1 for _,_,w,h in lane_divider_bboxes])
            frame_data.frame_orientation = FrameDataConst.HORIZONTAL if direction >= 0 else FrameDataConst.VERTICAL
            
        # call models 
        track_results = self.pose_caller.track_keypoints(frame, **self.pose_config['inference'])
        frame_keypoints = track_results.keypoints.xy
        # if nothing detected, its shape is (N,0,51)
        if frame_keypoints.shape[1] != 0:
            valid_skeletons, valid_swimmer_ids = get_valid_skeletons_from_track(track_results) # get multiple valid skeletons
        
        if  frame_keypoints.shape[1] != 0 and valid_skeletons.shape[0] != 0: # at least one valid skeleton
            frame_data.skeleton_list = valid_skeletons
            frame_data.swimmer_id_list = valid_swimmer_ids.numpy().astype(int)
            frame_data.bbox_list = [get_bbox(skel) for skel in frame_data.skeleton_list] # x y w h
            frame_data.bbox_area_list = [bbox[2] * bbox[3] for bbox in frame_data.bbox_list]

            if frame_idx % (10*FREQ_SEGMENT) == 0: # do the following every 10*FREQ_SEGMENT frames
                self.convert_pixel_to_meter(lane_divider_bboxes, frame_data.frame_orientation, frame_keypoints)

            # bbox_ground = get_ground(frame) # x y w h get ground to compute start and end time
            # if bbox_ground[0] != -1 and frame_data.bbox[0] != -1:
                # overlap = bbox_overlap_or_near(frame_data.bbox, bbox_ground, threshold=20)
            # else:
                # overlap = False

            # TODO: get face (can be optimised)
            # face_bboxes = self.face_caller.get_face(frame, **self.face_config['inference'])
            # face_bbox = self.stroke_process.match_face(frame_data.skeleton, face_bboxes)
            # frame_data.face_up = face_bbox != None
            frame_data.face_up_list = [False] * valid_skeletons.shape[0]
           

            # get skeleton direction
            frame_data.direction_list = [self.direction_process.get_skeleton_direction([skel]) for skel in frame_data.skeleton_list]
            
            # detect status only RACE/STOP now
            # frame_data.status = self.status_process.get_status(frame_data.skeleton, frame_data, self.frame_data_list,
                                                                # previous_interval=self.fps*3)
            
            # count stroke
            frame_data.stroke_count_list = []
            for swimmer_id in frame_data.swimmer_id_list:
                for i in range(len(self.frame_data_list)-1, -1, TIME_WINDOW_STROKE-1):
                    temp_frame_data_list = []
                    if swimmer_id in self.frame_data_list[i].swimmer_id_list:
                        temp_frame_data_list.append(self.frame_data_list[i])
                        frame_data.stroke_count_list.append(self.stroke_process.count_stroke(temp_frame_data_list))
                        
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

                # update previous anchor points and add projected points
                self.anchor_process.update_anchor_points(frame_idx, frame, self.prev_frame, frame_data, lane_divider_bboxes)

                # calculate speed
                for swimmer_id in frame_data.swimmer_id_list:
                    self.speed_process.calculate_speed(frame, frame_data, swimmer_id,
                                                    self.anchor_process.anchor_list,
                                                    lane_divider_bboxes) # return speed in pixels
                
                frame_data.speed_px_list = []
                frame_data.speed_m_list = []
                frame_data.speed_pct_change_list = []
                for swimmer_id in frame_data.swimmer_id_list:
                    if swimmer_id in self.speed_process.current_speed:
                        frame_data.speed_px_list.append(self.speed_process.current_speed[swimmer_id])
                        frame_data.speed_m_list.append(self.speed_process.current_speed[swimmer_id]/np.mean(self.pixel_to_meters)) # convert to meters
                        if swimmer_id in self.speed_process.pct_change:
                            frame_data.speed_pct_change_list.append(self.speed_process.pct_change[swimmer_id])
                        else:
                            frame_data.speed_pct_change_list.append(0)
                    else:
                        frame_data.speed_px_list.append(0)
                        frame_data.speed_m_list.append(0)
                        frame_data.speed_pct_change_list.append(0)

                # frame_data.speed_px_list = [self.speed_process.current_speed[swimmer_id] for swimmer_id in frame_data.swimmer_id_list]
                # frame_data.speed_m_list = [speed/np.mean(self.pixel_to_meters) for speed in frame_data.speed_px_list] # convert to meters
                # frame_data.speed_pct_change_list = [self.speed_process.pct_change[swimmer_id] for swimmer_id in frame_data.swimmer_id_list]
                # frame_data.red_marker = self.speed_process.red_marker
                updated_speed = True

            else:
                self.anchor_process.update_anchor_points(frame_idx, frame, self.prev_frame, frame_data, [])
                frame_data.speed_m_list = self.frame_data_list[-1].speed_m_list.copy()
                frame_data.speed_px_list = self.frame_data_list[-1].speed_px_list.copy()
                frame_data.speed_pct_change_list = self.frame_data_list[-1].speed_pct_change_list.copy()

        else:
            self.anchor_process.update_anchor_points(frame_idx, frame, self.prev_frame, frame_data, [])
            if self.frame_data_list:
                frame_data.speed_m_list = self.frame_data_list[-1].speed_m_list.copy()
                frame_data.speed_px_list = self.frame_data_list[-1].speed_px_list.copy()
                frame_data.speed_pct_change_list = self.frame_data_list[-1].speed_pct_change_list.copy()

        self.prev_frame = frame
        
                        
            
        if torch.is_tensor(frame_data.skeleton_list):
            frame_data.skeleton_list = frame_data.skeleton_list.cpu().numpy().tolist()
        # append current frame to list
        self.frame_data_list.append(frame_data)

        annotated_frame = frame.copy()
        annotated_frame = draw_keypoints(annotated_frame, frame_data.skeleton_list, thickness=1)
        annotated_frame = draw_detection(annotated_frame, lane_divider_bboxes)

        if debug:
            if len(bbox_ground) != 0 and bbox_ground[0] != -1:
                annotated_frame = draw_detection(annotated_frame, [bbox_ground])
            texts = frame_data.__str__()
            annotated_frame = write_texts(annotated_frame, texts, 30, org=(30,30))

        
        # draw anchor points    
        for k,v in self.anchor_process.anchor_list.items():
            if debug:
                frame_idx_ = str(k)
            else:
                frame_idx_ = ''
            point_list = np.array([ap.coord for ap in v])
            swimmer_ids = np.array([ap.swimmer_id for ap in v])
            for point, swimmer_id in zip(point_list, swimmer_ids):
                annotated_frame = draw_dot(annotated_frame, [point], 
                                            color=self.RANDOM_COLORS[swimmer_id%len(self.RANDOM_COLORS)].tolist(),
                                            radius=5,
                                            frame_idx=frame_idx_)
        
        # visualize swimmer id and speed
        for i in range(len(frame_data.swimmer_id_list)):
            swimmer_id = frame_data.swimmer_id_list[i]
            skel = frame_data.skeleton_list[i]
            spd = frame_data.speed_m_list[i] if len(frame_data.speed_m_list) > i else 0
            annotated_txt = f'ID:{swimmer_id} Speed:{spd:.2f}m/s'
            annotated_frame = write_texts(annotated_frame, [annotated_txt], 10, 
                                          org=(int(skel[0][0]), int(skel[0][1])),
                                          font_scale=0.5, color=self.RANDOM_COLORS[swimmer_id%len(self.RANDOM_COLORS)].tolist()
                                          )
            
        if len(self.frame_data_list) > self.fps * SAVE_AFTER_SECONDS:
            self.frame_data_list.pop(0)
            
        return annotated_frame, frame_data, updated_speed, overlap
    
