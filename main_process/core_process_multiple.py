from config.general import POSE_CONFIG, FACE_CONFIG, SEG_CONFIG, DETECT_CONFIG, SAVE_AFTER_SECONDS, NO_POINTS_SEGMENTATION, FREQ_SEGMENT, PIXEL_DENSITY_WIDTH, PIXEL_DENSITY_HEIGHT, TIME_WINDOW_STROKE, FPS_RATE, ID_TRACKER_WINDOW
from model_caller.pose_model_caller import PoseCallerYOLO
from model_caller.seg_model_caller import SegCallerYOLO
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

        # self.RANDOM_COLORS = np.random.randint(0, 255, (100, 3))
        self.RANDOM_COLORS = generate_even_light_colors(n=30)

        self.pose_config = read_yaml(POSE_CONFIG)
        self.face_config = read_yaml(FACE_CONFIG)
        self.seg_config = read_yaml(SEG_CONFIG)
        self.detection_config = read_yaml(DETECT_CONFIG)


        self.pose_caller = PoseCallerYOLO(self.pose_config['model_path'])
        self.seg_caller = SegCallerYOLO(self.seg_config['model_path'])


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

        self.prev_frame = np.array([])


        
    def swimming_calculation(self, frame:np.array, frame_idx:int, debug:bool=True) -> tuple:
        init_time = time.perf_counter()
        if len(self.prev_frame) == 0:
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
            frame_data.swimmer_id_list = self.frame_data_list[-1].swimmer_id_list.copy()
        
        frame_data.frame_idx = frame_idx
        lane_divider_bboxes = []
        bbox_ground = []

        # get full lane divider for frame orientation
        if frame_idx % (10*FREQ_SEGMENT) == 0: # do the following every 10*FREQ_SEGMENT frames
            start_time = time.perf_counter()
            lane_divider_bboxes = self.seg_caller.get_lane_dividers(frame, **self.seg_config['inference'])
            frame_data.segment_time = time.perf_counter() - start_time
            direction = sum([1 if w > h else -1 for _,_,w,h in lane_divider_bboxes])
            frame_data.frame_orientation = FrameDataConst.HORIZONTAL if direction >= 0 else FrameDataConst.VERTICAL
            
        # call models 
        start_time = time.perf_counter()
        # track_results = self.pose_caller.track_keypoints(frame, **self.pose_config['inference'])
        frame_keypoints = self.pose_caller.get_keypoints(frame, **self.pose_config['inference'])
        frame_data.pose_time = time.perf_counter() - start_time

        # frame_keypoints = track_results.keypoints.xy
        # if nothing detected, its shape is (N,0,51)
        if frame_keypoints.shape[1] != 0:
            # valid_skeletons, valid_swimmer_ids = get_valid_skeletons_from_track(track_results) # get multiple valid skeletons
            valid_skeletons = get_valid_skeletons(frame_keypoints)
            valid_swimmer_ids = torch.arange(valid_skeletons.shape[0])
        
        if  frame_keypoints.shape[1] != 0 and valid_skeletons.shape[0] != 0: # at least one valid skeleton
            valid_bboxes = [get_bbox(skel) for skel in valid_skeletons]
            valid_swimmer_ids = valid_swimmer_ids.numpy().astype(int)
            start_time = time.perf_counter()
            valid_swimmer_ids = self.tracker.reassign_swimmer_id(valid_skeletons, valid_swimmer_ids,
                                                                 [_frame_data.skeleton_list for _frame_data in self.frame_data_list[-ID_TRACKER_WINDOW:]],
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
            start_time = time.perf_counter()
            if len(self.frame_data_list) >= TIME_WINDOW_STROKE:
                frame_data.stroke_rate_list = []
                for swimmer_id in frame_data.swimmer_id_list:
                    temp_frame_data_list = []
                    for j in range(len(self.frame_data_list)-1, len(self.frame_data_list)-TIME_WINDOW_STROKE-1, -1):
                        if swimmer_id in self.frame_data_list[j].swimmer_id_list:
                            idx = np.where(np.array(self.frame_data_list[j].swimmer_id_list) == swimmer_id)[0][0]
                            temp_frame_data_list.append(self.frame_data_list[j].skeleton_list[idx])
                    if temp_frame_data_list:
                        stroke_rate = self.stroke_process.count_stroke(np.array(temp_frame_data_list))
                        frame_data.stroke_rate_list.append(stroke_rate)
                    else:
                        frame_data.stroke_rate_list.append(0)
            else:
                frame_data.stroke_rate_list = [0] * len(frame_data.swimmer_id_list)
            frame_data.count_stroke_time = time.perf_counter() - start_time

                        
            # TODO: classify stroke
            # frame_data.stroke = self.stroke_process.classify_stroke(frame_data, self.frame_data_list)
            
            # TODO: fix left right swap
            # frame_data.skeleton = side_process.get_correct_side(frame_data.skeleton, frame_data)
                
            if frame_idx % FREQ_SEGMENT == 0: # calculate instantaneous speed every FREQ_SEGMENT frames
                if len(lane_divider_bboxes) == 0:
                    start_time = time.perf_counter()
                    lane_divider_bboxes = self.seg_caller.get_lane_dividers(frame, **self.seg_config['inference'])
                    frame_data.segment_time = time.perf_counter() - start_time
                
                # generate random points inside the lane
                start_time = time.perf_counter()
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
                frame_data.anchor_update_time = time.perf_counter() - start_time

                # calculate speed
                start_time = time.perf_counter()
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
                        if len(self.pixel_to_meters) > 0:
                            frame_data.speed_m_list.append(self.speed_process.current_speed[swimmer_id]/np.mean(self.pixel_to_meters)) # convert to meters
                        else:
                            frame_data.speed_m_list.append(0)
                        if swimmer_id in self.speed_process.pct_change:
                            frame_data.speed_pct_change_list.append(self.speed_process.pct_change[swimmer_id])
                        else:
                            frame_data.speed_pct_change_list.append(0)
                    else:
                        frame_data.speed_px_list.append(0)
                        frame_data.speed_m_list.append(0)
                        frame_data.speed_pct_change_list.append(0)
                updated_speed = True
                frame_data.speed_calculation_time = time.perf_counter() - start_time

                # Calculate distance per stroke
                # distance = speed * time, time = TIME_WINDOW_STROKE/(self.fps//FPS_RATE)
                # this time is used to count strokes in stroke_process
                # convert stroke count to 3s
                # distance per stroke = distance / stroke count (in 3s)
                frame_data.distance_per_stroke_list = []
                for i in range(len(frame_data.swimmer_id_list)):
                    distance_swum = frame_data.speed_m_list[i] 
                    stroke_rate = frame_data.stroke_rate_list[i]
                    distance_per_stroke = distance_swum/stroke_rate * 60 if stroke_rate != 0 else 0
                    frame_data.distance_per_stroke_list.append(round(distance_per_stroke,2))

            else:
                start_time = time.perf_counter()
                self.anchor_process.update_anchor_points(frame_idx, frame, self.prev_frame, frame_data, [])
                frame_data.anchor_update_time = time.perf_counter() - start_time

                # There are cases when the skeletons are detected but the speed is not updated
                # In that case, len of skeletons and speed list will not be the same
                frame_data.speed_m_list = self.frame_data_list[-1].speed_m_list.copy()
                frame_data.speed_px_list = self.frame_data_list[-1].speed_px_list.copy()
                frame_data.speed_pct_change_list = self.frame_data_list[-1].speed_pct_change_list.copy()
                frame_data.distance_per_stroke_list = self.frame_data_list[-1].distance_per_stroke_list.copy()
                if len(frame_data.speed_m_list) < len(frame_data.skeleton_list): # add 0 to the end of speed list if new swimmer appears
                    offset = len(frame_data.skeleton_list) - len(frame_data.speed_m_list)
                    frame_data.speed_m_list.extend([0]*offset)
                    frame_data.speed_px_list.extend([0]*offset)
                    frame_data.speed_pct_change_list.extend([0]*offset)
                    frame_data.distance_per_stroke_list.extend([0]*offset)

        else:
            start_time = time.perf_counter()
            self.anchor_process.update_anchor_points(frame_idx, frame, self.prev_frame, frame_data, [])
            frame_data.anchor_update_time = time.perf_counter() - start_time
            if self.frame_data_list:
                frame_data.speed_m_list = self.frame_data_list[-1].speed_m_list.copy()
                frame_data.speed_px_list = self.frame_data_list[-1].speed_px_list.copy()
                frame_data.speed_pct_change_list = self.frame_data_list[-1].speed_pct_change_list.copy()
                frame_data.distance_per_stroke_list = self.frame_data_list[-1].distance_per_stroke_list.copy()
                frame_data.stroke_rate_list = self.frame_data_list[-1].stroke_rate_list.copy()

        self.prev_frame = frame
        
                        
        start_time = time.perf_counter()
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
        
        # visualize swimmer id, speed and stroke count
        for i in range(len(frame_data.swimmer_id_list)):
            swimmer_id = frame_data.swimmer_id_list[i]

            skel = frame_data.skeleton_list[i]
            spd = frame_data.speed_m_list[i] 
            stroke_rate = frame_data.stroke_rate_list[i]
            distance_per_stroke = frame_data.distance_per_stroke_list[i]

            annotated_txt = f'ID:{swimmer_id}, {spd:.2f}m/s, {stroke_rate} spm, {distance_per_stroke} dps'
            annotated_frame = write_texts(annotated_frame, [annotated_txt], 10, 
                                          org=(int(skel[0][0]), int(skel[0][1])),
                                          font_scale=0.5, color=self.RANDOM_COLORS[swimmer_id%len(self.RANDOM_COLORS)].tolist()
                                          )
        frame_data.visualization_time = time.perf_counter() - start_time
            
        if len(self.frame_data_list) > self.fps * SAVE_AFTER_SECONDS:
            self.frame_data_list.pop(0)
        frame_data.total_time = time.perf_counter() - init_time
            
        return annotated_frame, frame_data, updated_speed, overlap
    
