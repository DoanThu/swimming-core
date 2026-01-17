from config.general import POSE_CONFIG, FACE_CONFIG, SEG_CONFIG, DETECT_CONFIG, SAVE_AFTER_SECONDS, NO_POINTS_SEGMENTATION, FREQ_SEGMENT, PIXEL_DENSITY_WIDTH, PIXEL_DENSITY_HEIGHT, TIME_WINDOW_STROKE, FPS_RATE, ID_TRACKER_WINDOW
from model_caller.pose_model_caller import PoseCallerYOLO, PoseWorker
from model_caller.seg_model_caller import SegCallerYOLO, SegWorker
from model_caller.track_caller import TrackCallerSkeleton
import torch 
import cv2 
from utils.lane_segment_utils import get_ground, bbox_overlap_or_near
from utils.visualize_utils import draw_keypoints, write_texts, draw_segmentation, draw_dot, draw_detection, generate_even_light_colors
from utils.file_utils import read_yaml
from utils.skeleton_utils import get_valid_skeletons_from_track, get_bbox, get_valid_skeletons
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
import numpy as np
import time
import logging 
import queue
from postprocess.visualization_process import visualize_results, visualize_swimmer_id
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
        if not distances: return
        temp = np.mean(distances)/2.5*PIXEL_DENSITY_HEIGHT/PIXEL_DENSITY_WIDTH
        if temp == 0: return

        if len(self.pixel_to_meters) == 0:
            self.pixel_to_meters.append(temp)
        else:
            if abs(temp - np.mean(self.pixel_to_meters))/temp < 0.1:
                self.pixel_to_meters.append(temp)
        if len(self.pixel_to_meters) > 20:
            self.pixel_to_meters.pop(0)

    def __init__(self, fps:int):
        self.pixel_to_meters = []

        self.fps = fps # not considered FPS_RATE here
        self.frame_data_list = [] # list of FrameMultipleData
        self.prev_frame = None # store previous image 
        # self.raw_results_from_model = [] # store raw results from model for cpu processing

        # self.RANDOM_COLORS = generate_even_light_colors(n=30)

        self.pose_config = read_yaml(POSE_CONFIG)
        self.seg_config = read_yaml(SEG_CONFIG)


        # self.pose_caller = PoseCallerYOLO(self.pose_config['model_path'])
        # self.seg_caller = SegCallerYOLO(self.seg_config['model_path'])

        
        # # Initialize persistent threads
        # self.pose_in_q, self.pose_out_q = queue.Queue(maxsize=4), queue.Queue()
        # self.seg_in_q,  self.seg_out_q  = queue.Queue(maxsize=4), queue.Queue()

        # self.pose_thread = PoseWorker(self.pose_config['model_path'], self.pose_in_q, self.pose_out_q)
        # self.seg_thread  = SegWorker(self.seg_config['model_path'],  self.seg_in_q,  self.seg_out_q)
        
        # # Start persistent workers
        # self.pose_thread.start()
        # self.seg_thread.start()


        if torch.cuda.is_available():
            logging.info('CUDA is available. Loading pose model from ' + self.pose_config['model_path'])
            logging.info('CUDA is available. Loading segmentation model from ' + self.seg_config['model_path'])

        else:
            logging.info('CUDA is not available. Using CPU instead.')


        self.stroke_process = StrokeProcess(fps=self.fps)
        self.status_process = StatusProcess()
        self.side_process = SideProcess()
        self.direction_process = DirectionProcess()
        self.speed_process = SpeedProcess(fps=self.fps)
        self.anchor_process = AnchorProcess(window=self.fps*5)
        self.lane_divider_process = LaneDivider()
        self.tracker = TrackCallerSkeleton(window=ID_TRACKER_WINDOW)



        
    def swimming_calculation(self, frame:np.array, frame_idx:int, 
                            frame_keypoints:torch.Tensor, lane_divider_bboxes:List,
                            debug:bool=True, from_socket:bool=False) -> tuple:
        init_time = time.perf_counter()
        # if len(self.prev_frame) == 0:
        if self.prev_frame is None:
            self.prev_frame = frame

        updated_speed = False 
        overlap = False

        frame_data = FrameMultipleData()
        frame_data.swimmer_id_list = []
        frame_data.skeleton_list = []
    
        if len(self.frame_data_list) != 0:
            frame_data.status_list = self.frame_data_list[-1].status_list.copy()
            frame_data.frame_orientation = self.frame_data_list[-1].frame_orientation
            frame_data.skeleton_list = self.frame_data_list[-1].skeleton_list.copy()
            
            prev_ids = self.frame_data_list[-1].swimmer_id_list
            if isinstance(prev_ids, torch.Tensor):
                frame_data.swimmer_id_list = prev_ids.clone()
            elif isinstance(prev_ids, np.ndarray):
                frame_data.swimmer_id_list = prev_ids.copy()
            else:
                frame_data.swimmer_id_list = list(prev_ids)
        
        bbox_ground = []

        # Process previous frame while waiting for current frame results
        frame_data.frame_idx = frame_idx
        direction = sum([1 if w > h else -1 for _,_,w,h in lane_divider_bboxes])
        frame_data.frame_orientation = FrameDataConst.HORIZONTAL if direction >= 0 else FrameDataConst.VERTICAL

        # if nothing detected, its shape is (N,0,51)
        if frame_keypoints.shape[1] != 0:
            valid_skeletons = get_valid_skeletons(frame_keypoints)
            valid_swimmer_ids = torch.arange(valid_skeletons.shape[0])
        if  frame_keypoints.shape[1] != 0 and valid_skeletons.shape[0] != 0: # at least one valid skeleton
            valid_bboxes = [get_bbox(skel) for skel in valid_skeletons]
            start_time = time.perf_counter()
            # Keep IDs as tensors for faster processing
            valid_swimmer_ids = self.tracker.reassign_swimmer_id_v2(valid_skeletons, valid_swimmer_ids,
                                                                [getattr(_frame_data, 'skeleton_tensor', _frame_data.skeleton_list) 
                                                                 for _frame_data in self.frame_data_list[-ID_TRACKER_WINDOW:]],
                                                                [_frame_data.swimmer_id_list for _frame_data in self.frame_data_list[-ID_TRACKER_WINDOW:]],
                                                                frame_data.frame_orientation
                                                                )
            frame_data.tracking_time = time.perf_counter() - start_time

            frame_data.skeleton_list = valid_skeletons
            frame_data.swimmer_id_list = valid_swimmer_ids
            frame_data.bbox_list = valid_bboxes # x y w h
            frame_data.bbox_area_list = [bbox[2] * bbox[3] for bbox in frame_data.bbox_list]

            
            if frame_idx % (10*FREQ_SEGMENT) == 0: # do the following every 10*FREQ_SEGMENT frames
                self.convert_pixel_to_meter(lane_divider_bboxes, frame_data.frame_orientation, frame_keypoints)

            frame_data.face_up_list = [False] * valid_skeletons.shape[0]

            # get skeleton direction
            frame_data.direction_list = [self.direction_process.get_skeleton_direction(skel) for skel in frame_data.skeleton_list]
            
            # count stroke
            start_time = time.perf_counter()
            if len(self.frame_data_list) >= TIME_WINDOW_STROKE:
                frame_data.stroke_count_list = []
                for swimmer_id in frame_data.swimmer_id_list:
                    temp_frame_data_list = []
                    for j in range(len(self.frame_data_list)-1, len(self.frame_data_list)-TIME_WINDOW_STROKE-1, -1):
                        try:
                            idx = np.where(np.array(self.frame_data_list[j].swimmer_id_list) == swimmer_id)[0][0]
                        except Exception as e:
                            continue
                        temp_frame_data_list.append(self.frame_data_list[j].skeleton_list[idx])
                    if temp_frame_data_list:
                        strk_count = self.stroke_process.count_stroke(temp_frame_data_list)
                    else:
                        strk_count = 0
                    
                    if strk_count == 0 and len(self.frame_data_list) > 0:
                        prev_ids = self.frame_data_list[-1].swimmer_id_list
                        if isinstance(prev_ids, torch.Tensor): prev_ids = prev_ids.tolist()
                        elif isinstance(prev_ids, np.ndarray): prev_ids = prev_ids.tolist()
                        
                        s_id = int(swimmer_id)
                        if s_id in prev_ids:
                            idx = prev_ids.index(s_id)
                            strk_count = self.frame_data_list[-1].stroke_count_list[idx]
                    frame_data.stroke_count_list.append(strk_count)
            else:
                frame_data.stroke_count_list = [0] * len(frame_data.swimmer_id_list)
            # frame_data.stroke_count_list = [0] * len(frame_data.swimmer_id_list)

            frame_data.count_stroke_time = time.perf_counter() - start_time

            # TODO: classify stroke
            # TODO: fix left right swap

            if frame_idx % FREQ_SEGMENT == 0: # calculate instantaneous speed every FREQ_SEGMENT frames
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
                start_time = time.perf_counter()
                self.anchor_process.update_anchor_points(frame_idx, frame, self.prev_frame, frame_data, lane_divider_bboxes)
                frame_data.anchor_update_time = time.perf_counter() - start_time

                # calculate speed
                start_time = time.perf_counter()
                for swimmer_id in frame_data.swimmer_id_list:
                    self.speed_process.calculate_speed(frame, frame_data, swimmer_id,
                                                    self.anchor_process.anchor_list,
                                                    lane_divider_bboxes) # return speed in pixels
                updated_speed = True
                frame_data.speed_calculation_time = time.perf_counter() - start_time

            else:
                start_time = time.perf_counter()
                self.anchor_process.update_anchor_points(frame_idx, frame, self.prev_frame, frame_data, [])
                frame_data.anchor_update_time = time.perf_counter() - start_time

            # Construct speed lists (Always)
            frame_data.speed_px_list = []
            frame_data.speed_m_list = []
            frame_data.speed_pct_change_list = []
            
            for swimmer_id in frame_data.swimmer_id_list:
                s_id = int(swimmer_id)
                if s_id in self.speed_process.current_speed:
                    px = self.speed_process.current_speed[s_id]
                    frame_data.speed_px_list.append(px)
                    if len(self.pixel_to_meters) > 0:
                        frame_data.speed_m_list.append(px/np.mean(self.pixel_to_meters))
                    else:
                        frame_data.speed_m_list.append(0)
                    
                    if s_id in self.speed_process.pct_change:
                        frame_data.speed_pct_change_list.append(self.speed_process.pct_change[s_id])
                    else:
                        frame_data.speed_pct_change_list.append(0)
                else:
                    frame_data.speed_px_list.append(0)
                    frame_data.speed_m_list.append(0)
                    frame_data.speed_pct_change_list.append(0)

            # Calculate DPS (Always)
            frame_data.distance_per_stroke_list = []
            for i in range(len(frame_data.swimmer_id_list)):
                distance_swum = frame_data.speed_m_list[i] 
                stroke_count = frame_data.stroke_count_list[i]
                distance_per_stroke = distance_swum/stroke_count * 60 if stroke_count != 0 else 0
                frame_data.distance_per_stroke_list.append(round(distance_per_stroke,2))

        else:
            start_time = time.perf_counter()
            self.anchor_process.update_anchor_points(frame_idx, frame, self.prev_frame, frame_data, [])
            frame_data.anchor_update_time = time.perf_counter() - start_time
            if self.frame_data_list:
                frame_data.speed_m_list = self.frame_data_list[-1].speed_m_list.copy()
                frame_data.speed_px_list = self.frame_data_list[-1].speed_px_list.copy()
                frame_data.speed_pct_change_list = self.frame_data_list[-1].speed_pct_change_list.copy()
                frame_data.distance_per_stroke_list = self.frame_data_list[-1].distance_per_stroke_list.copy()
                frame_data.stroke_count_list = self.frame_data_list[-1].stroke_count_list.copy()

        if torch.is_tensor(frame_data.skeleton_list):
            frame_data.skeleton_list = frame_data.skeleton_list.cpu().numpy().tolist()

        # append current frame to list
        self.frame_data_list.append(frame_data)

        # Visualization
        start_time = time.perf_counter()
        annotated_frame = visualize_results(frame, frame_data, lane_divider_bboxes,bbox_ground, self.anchor_process.anchor_list, debug, from_socket=from_socket)
        frame_data.visualization_time = time.perf_counter() - start_time
            
        if len(self.frame_data_list) > self.fps * SAVE_AFTER_SECONDS:
            self.frame_data_list.pop(0)


        self.prev_frame = frame
        frame_data.analysis_time = time.perf_counter() - init_time
        return annotated_frame, frame_data, updated_speed, overlap

    
