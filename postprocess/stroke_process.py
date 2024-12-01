import numpy as np
import torch 
from typing import List
from postprocess.data import FrameData, FrameDataConst

class StrokeProcess:
    def __init__(self, time_window=180) -> None:
        self.time_window = time_window
    
    def match_face(self, swimmer_skeleton: torch.Tensor, face_bboxes: torch.Tensor) -> torch.Tensor:
        """ Return face that belongs to the skeleton

        Args:
            swimmer_skeleton (torch.Tensor): shape=(1,17,2)
            face_bboxes (torch.Tensor): shape=(N,4)

        Returns:
            torch.Tensor: return the bounding box of the face, shape=(1,4)
        """

        def get_center(xmin, ymin, xmax, ymax):
            return (xmin+xmax)/2, (ymin+ymax)/2
        
        def dist(p1: torch.Tensor, p2: torch.Tensor) -> float:
            p1 = [_p.cpu().data.numpy() for _p in p1] 
            p2 = [_p.cpu().data.numpy() for _p in p2] 
            return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
        
        def get_threshold(points) -> float:
            xmin, ymin = points.min(axis=0).values
            xmax, ymax = points.max(axis=0).values
            return max(ymax-ymin,xmax-xmin)
        
        nose = swimmer_skeleton[0][0]
        dist_thres = get_threshold(swimmer_skeleton[0])
        centers = [get_center(*bbox) for bbox in face_bboxes]
        dist_arr = [dist(center, nose) for center in centers]
        
        if len(dist_arr) == 0: return 
        
        box_idx = np.argmin(dist_arr)
        if dist_arr[box_idx] > dist_thres:
            return
        return face_bboxes[box_idx]
    
    def classify_stroke(self, frame_data: FrameData, frame_data_list: List[FrameData], threshold=0.8) -> int:
        """ classify stroke given the info of previous frames. Now can detect backstroke only.

        Args:
            frame_data (FrameData): current frame info
            frame_data_list (List[FrameData]): info of previous frames
            threshold (float, optional): percentage of facing up during certain time window. Defaults to 0.8.

        Returns:
            int: return the stroke 
        """
        stroke = FrameDataConst.UNKNOWN
        direction = frame_data.direction
        count_face = [1 for _frame in frame_data_list[-self.time_window:] if _frame.face_up and _frame.direction == direction]
        if sum(count_face)/self.time_window > threshold: # likely to be backstroke
            stroke = FrameDataConst.BACKSTROKE
            
        return stroke
                