from typing import List
from postprocess.data import FrameData, FrameDataConst
import numpy as np
import torch


class SideProcess:
    def __init__(self) -> None:
        pass
    
    def swap_side(self, skeleton: torch.Tensor) -> torch.Tensor:
        """ swap two sides of the skeleton

        Args:
            skeleton (torch.Tensor): shape = (1, 17, 2)

        Returns:
            torch.Tensor: swapped side skeleton, shape = (1, 17, 2)
        """
        from copy import deepcopy
        swapped_skeleton = deepcopy(skeleton)
        swapped_skeleton[0][1:16:2] = skeleton[0][2:17:2]
        swapped_skeleton[0][2:17:2] = skeleton[0][1:16:2]
        return swapped_skeleton
    
    
    
    def get_correct_side(self, skeleton: torch.Tensor, frame_data: FrameData) -> torch.Tensor:
        """ Swap two sides if they are not correct, return swapped skeleton

        Args:
            skeleton (torch.Tensor): shape = (1, 17, 2)
            frame_data (FrameData): info of the current frame

        Returns:
            torch.Tensor: return skeleton with correct sides, shape = (1, 17, 2)
        """
        left_wrist = skeleton[0][9]
        right_wrist = skeleton[0][10]
        if frame_data.stroke == FrameDataConst.BACKSTROKE: 
            if frame_data.direction == FrameDataConst.UP and left_wrist[0] < right_wrist[0]:
                skeleton = self.swap_side(skeleton)
            elif frame_data.direction == FrameDataConst.DOWN and left_wrist[0] > right_wrist[0]:
                skeleton = self.swap_side(skeleton)
            elif frame_data.direction == FrameDataConst.LEFT and left_wrist[1] > right_wrist[0]:
                skeleton = self.swap_side(skeleton)
            elif frame_data.direction == FrameDataConst.RIGHT and left_wrist[1] < right_wrist[0]:
                skeleton = self.swap_side(skeleton)
        else:
            if frame_data.direction == FrameDataConst.UP and left_wrist[0] > right_wrist[0]:
                skeleton = self.swap_side(skeleton)
            elif frame_data.direction == FrameDataConst.DOWN and left_wrist[0] < right_wrist[0]:
                skeleton = self.swap_side(skeleton)
            elif frame_data.direction == FrameDataConst.LEFT and left_wrist[1] < right_wrist[0]:
                skeleton = self.swap_side(skeleton)
            elif frame_data.direction == FrameDataConst.RIGHT and left_wrist[1] > right_wrist[0]:
                skeleton = self.swap_side(skeleton) 
        return skeleton
        