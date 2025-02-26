from typing import List
from postprocess.data import FrameData, FrameDataConst
import torch
import math
import numpy as np
class StatusProcess:
    def __init__(self, window_size=60):
        self.left_side_length_arr = np.array([])
        self.wrist_distance_arr = np.array([])
        self.leg_over_shoulder_arr = np.array([])
        self.window_size = window_size
    
    def get_status(self, skeleton: torch.Tensor, frame_data: FrameData, frame_data_list: List[FrameData],
                    previous_interval:int=60, threshold:float=0.8) -> int:
        """ Get status of the skeleton in the current frame

        Args:
            skeleton (torch.Tensor): skeleton detected in the frame, size=[1,17,2]
            frame_data (FrameData): metadata of the current frame
            frame_data_list (List[FrameData]): list of previous frames' metadata
            previous_interval (int, optional): time interval to lookback. Defaults to 60.
            threshold (float, optional): threshold percentage to decide the current status. Defaults to 0.8.

        Returns:
            int: number that represents the status of the skeleton
        """
        def get_length(p1: np.ndarray, p2: np.ndarray) -> float:
            """ Compute length between two 2-D points

            Args:
                p1 (np.ndarray): first point, size = [1,2]
                p2 (np.ndarray): second point, size = [1,2]

            Returns:
                float: distance between 2 input points
            """
            return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        
        left_hip, left_knee, left_ankle = skeleton[0][11].cpu().numpy(), skeleton[0][13].cpu().numpy(), skeleton[0][15].cpu().numpy()
        right_hip, right_knee, right_ankle = skeleton[0][12].cpu().numpy(), skeleton[0][14].cpu().numpy(), skeleton[0][16].cpu().numpy()
        left_leg_length = get_length(left_hip, left_knee) + get_length(left_knee, left_ankle)
        right_leg_length = get_length(right_hip, right_knee) + get_length(right_knee, right_ankle)
        leg_length = max(left_leg_length, right_leg_length)
        
        left_shoulder, right_shoulder = skeleton[0][5].cpu().numpy(), skeleton[0][6].cpu().numpy()
        shoulder_length = get_length(left_shoulder, right_shoulder)
        left_side_length = get_length(left_shoulder, left_hip)
        self.left_side_length_arr = np.append(self.left_side_length_arr, left_side_length)
        self.left_side_length_arr = self.left_side_length_arr[-self.window_size:]
        
        left_wrist, right_wrist = skeleton[0][9].cpu().numpy(), skeleton[0][10].cpu().numpy()
        wrist_distance = get_length(left_wrist, right_wrist)
        self.wrist_distance_arr = np.append(self.wrist_distance_arr, wrist_distance)
        self.wrist_distance_arr = self.wrist_distance_arr[-self.window_size:]
        
        
        ankle_distance = get_length(left_ankle, right_ankle)
        
        
        self.leg_over_shoulder_arr = np.append(self.leg_over_shoulder_arr, leg_length/shoulder_length >= 2)
        self.leg_over_shoulder_arr = self.leg_over_shoulder_arr[-self.window_size:]

        if leg_length/shoulder_length >= 1.5: return FrameDataConst.RACE
        return FrameDataConst.STOP

        
        # if frame_data.status == FrameDataConst.READY:
        #     if leg_length/shoulder_length >= 2: # can be "jump" 
        #         if np.mean(self.leg_over_shoulder_arr) >= threshold:
        #             return FrameDataConst.JUMP
        #     return frame_data.status
        
        # elif frame_data.status == FrameDataConst.JUMP:
        #     # can be "dolphin kick"
        #     if leg_length/shoulder_length >= 2: 
        #         if np.mean(self.wrist_distance_arr<ankle_distance) >= threshold:
        #             return FrameDataConst.DOLPHIN_KICK
        #     return frame_data.status
                
        # elif frame_data.status == FrameDataConst.DOLPHIN_KICK:
        #     # race
        #     if leg_length/shoulder_length >= 2: 
        #         if np.mean(self.wrist_distance_arr<ankle_distance) >= threshold:
        #             return FrameDataConst.DOLPHIN_KICK
        #     return FrameDataConst.RACE
        # elif frame_data.status == FrameDataConst.RACE:
        #     # dolphin kick 
        #     if leg_length/shoulder_length >= 2: 
        #         if np.mean(self.wrist_distance_arr<ankle_distance) >= threshold:
        #             return FrameDataConst.DOLPHIN_KICK
        #     # can be "turn"
        #     else:
        #         previous_status = [True if i.status == FrameDataConst.RACE or i.status == FrameDataConst.TURN else False for i in frame_data_list[-previous_interval:]]
        #         if np.mean(previous_status) >= threshold:
        #             return FrameDataConst.TURN
        #     return frame_data.status
            
        # elif frame_data.status == FrameDataConst.TURN:
        #     # can be "race" or "stop"
        #     previous_status = [True if i.status == FrameDataConst.TURN or i.status == FrameDataConst.STOP else False for i in frame_data_list[-previous_interval:]]
        #     if np.mean(previous_status) >= threshold:
        #         if leg_length/shoulder_length < 2:
        #             return FrameDataConst.STOP
        #         else:
        #             return FrameDataConst.RACE
        #     return frame_data.status
        
        # return frame_data.status
        