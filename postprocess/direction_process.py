import torch 
from postprocess.single_data import FrameDataConst
import numpy as np
import math


class DirectionProcess:
    def __init__(self) -> None:
        pass
    
    def get_skeleton_direction(self, skeleton: torch.Tensor) -> int:
        """ Return direction of the swimmer based on the detected skeleton.
        First, compute the angles between the vector between the mid shoulder and nose with Ox and Oy, to detect if the direction is towards Ox or Oy.
        Then left, right or up, down direction will be returned, respectively.

        Args:
            skeleton (torch.Tensor): skeleton shape = (1,17,2)

        Returns:
            int: one of four directions (left/right/up/down)
        """

        def length(x):
            return math.sqrt(x[0]**2 + x[1]**2)

        left_shoulder = skeleton[0][5].cpu().numpy()
        right_shoulder = skeleton[0][6].cpu().numpy()
        mid_shoulder = np.array([(left_shoulder[0]+right_shoulder[0])/2,
                                (left_shoulder[1]+right_shoulder[1])/2])
        
        nose = skeleton[0][0].cpu().numpy()
        nose_shoulder = np.array([nose[0]-mid_shoulder[0],nose[1]-mid_shoulder[1]])
        
        directions = np.array([[0,1],[1,0]]) # vertical and horizontal
        length_multiple = length(directions[0])*length(nose_shoulder)
        angles = []
        for direction in directions:
            angle = -math.degrees(math.asin((direction[0] * nose_shoulder[1] - direction[1] * nose_shoulder[0])/length_multiple))
            angle = abs(angle)
            angles.append(angle)
            
        min_idx = np.argmin(angles)
        if min_idx == 0: # vertical
            if nose[1] > mid_shoulder[1]:
                return FrameDataConst.DOWN
            return FrameDataConst.UP
        else: # horizontal
            if nose[0] > mid_shoulder[0]:
                return FrameDataConst.RIGHT
            return FrameDataConst.LEFT
        
        