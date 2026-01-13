import numpy as np
import torch 
from typing import List
from postprocess.single_data import FrameData, FrameDataConst
import math
from config.general import FPS_RATE
from scipy.signal import find_peaks, savgol_filter
from postprocess.analytics import ExtractParams
from config.general import TIME_WINDOW_STROKE
from postprocess.wave_peak_counter import count_wave_peaks_and_plot
from scipy.stats.mstats import winsorize

class StrokeProcess:
    def __init__(self, fps, time_window=TIME_WINDOW_STROKE) -> None:
        self.fps = fps
        self.time_window = time_window
    
    # def match_face(self, swimmer_skeleton: torch.Tensor, face_bboxes: torch.Tensor) -> torch.Tensor:
    #     """ Return face that belongs to the skeleton

    #     Args:
    #         swimmer_skeleton (torch.Tensor): shape=(1,17,2)
    #         face_bboxes (torch.Tensor): shape=(N,4)

    #     Returns:
    #         torch.Tensor: return the bounding box of the face, shape=(1,4)
    #     """

    #     def get_center(xmin, ymin, xmax, ymax):
    #         return (xmin+xmax)/2, (ymin+ymax)/2
        
    #     def dist(p1: torch.Tensor, p2: torch.Tensor) -> float:
    #         p1 = [_p.cpu().data.numpy() for _p in p1] 
    #         p2 = [_p.cpu().data.numpy() for _p in p2] 
    #         return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
        
    #     def get_threshold(points) -> float:
    #         xmin, ymin = points.min(axis=0).values
    #         xmax, ymax = points.max(axis=0).values
    #         return max(ymax-ymin,xmax-xmin)
        
    #     nose = swimmer_skeleton[0][0]
    #     dist_thres = get_threshold(swimmer_skeleton[0])
    #     centers = [get_center(*bbox) for bbox in face_bboxes]
    #     dist_arr = [dist(center, nose) for center in centers]
        
    #     if len(dist_arr) == 0: return 
        
    #     box_idx = np.argmin(dist_arr)
    #     if dist_arr[box_idx] > dist_thres:
    #         return
    #     return face_bboxes[box_idx]

    def get_length(self, p1: np.ndarray, p2: np.ndarray) -> float:
            """ Compute length between two 2-D points

            Args:
                p1 (np.ndarray): first point, size = [1,2]
                p2 (np.ndarray): second point, size = [1,2]

            Returns:
                float: distance between 2 input points
            """
            return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    def classify_stroke(self, frame_data: FrameData, frame_data_list: List[FrameData], 
                        threshold_status:float=0.8,
                        threshold_correlation:int=8) -> int:
        """ classify stroke given the info of previous frames. Now can detect backstroke only.

        Args:
            frame_data (FrameData): current frame info
            frame_data_list (List[FrameData]): info of previous frames
            threshold (float, optional): percentage of facing up during certain time window. Defaults to 0.8.

        Returns:
            int: return the stroke 
        """
        if len(frame_data_list) < self.time_window: return FrameDataConst.UNKNOWN
        overall_status = [1 if _frame.status != FrameDataConst.STOP else 0 for _frame in frame_data_list[-self.time_window:]]
        if np.mean(overall_status) < threshold_status: return FrameDataConst.UNKNOWN

        dist_lwrist_nose, dist_rwrist_nose = [], []
        for _frame in frame_data_list[-self.time_window:]:
            skeleton = _frame.skeleton
            left_wrist, right_wrist, nose = skeleton[0][9], skeleton[0][10], skeleton[0][0]
            dist_lwrist_nose.append(self.get_length(left_wrist, nose))
            dist_rwrist_nose.append(self.get_length(right_wrist, nose))

        correlation = np.correlate(dist_lwrist_nose, dist_rwrist_nose, 'full')
        argmax = np.argmax(correlation)
        if abs(len(correlation)//2-argmax) < threshold_correlation:
            return FrameDataConst.BUTTERFLY
        return FrameDataConst.FREESTYLE


    def count_stroke(self, skeleton_series: list) -> float:
        # skeleton_series: shape=(N,17,2), N is number of frames
        
        if len(skeleton_series) + 10 < self.time_window: 
            return 0 # not enough points
        
        # return 0 -> total time ~ 17ms

        dist_wrist_head = []
        extractParams = ExtractParams()

        for skeleton in skeleton_series[-self.time_window:]:
            if len(skeleton) == 0:
                continue
            left_wrist, head = skeleton[9], skeleton[0]
            real_distance = extractParams.distance(left_wrist[0], left_wrist[1], head[0], head[1])
            dist_wrist_head.append(real_distance) 

        if len(dist_wrist_head) + 10 < self.time_window: return 0 # not enough points

        # return 0 -> total time ~ 18ms
    
        dist_wrist_head = winsorize(np.array(dist_wrist_head), limits=[0.05, 0.05])
        # return 0 -> total time ~ 20ms
        dist_wrist_head = savgol_filter(dist_wrist_head, window_length=11, polyorder=3)

        stroke_no, peaks_idx, mask, diag = count_wave_peaks_and_plot(
                                                                dist_wrist_head, self.fps//FPS_RATE, 1, 0.5,
                                                                clf_kwargs=dict(min_freq=0.5, peak_ratio_thr=6.0, flatness_thr=0.55, ac_peak_thr=0.2),
                                                                min_prominence=0.3, debug=False
                                                            )
        # return 0 -> total time ~ 21ms
        if len(peaks_idx) > 1:
            diff_peaks = np.diff(peaks_idx)
            avg_diff = np.mean(diff_peaks)
            stroke_rate = 60/(avg_diff/(self.fps//FPS_RATE))
            return round(stroke_rate,2)

        duration_in_seconds = self.time_window/(self.fps//FPS_RATE)
        # print(f'len i peak = {len(i_peaks)}')
        # print(f'duration_in_seconds = {duration_in_seconds}')
        stroke_rate = stroke_no*60/duration_in_seconds
        return stroke_rate # stroke per minute
    
        # total time ~ 23ms 

