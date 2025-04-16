import torch
import numpy as np
from postprocess.data import FrameDataConst


def is_valid_skeleton(skeleton:torch.Tensor) -> bool:
    """ Return if the skeleton is valid.
    A valid skeleton is a skeleton that does not have any point at (0,0), which happens when the model cannot detect the pose.

    Args:
        skeleton (torch.Tensor): shape = (17,2)

    Returns:
        bool: return True if all 17 keypoints are valid
    """
    zero_tensor = torch.zeros(17,2).cuda()
    compare = torch.eq(zero_tensor, skeleton)
    return not torch.any(torch.all(compare, dim=1))


def is_valid_skeletons(skeletons:torch.Tensor) -> bool:
    """ Return if the list of skeleton is valid.
    A valid list has all valid skeletons.

    Args:
        skeletons (torch.Tensor): shape = (N,17,2)

    Returns:
        bool: return True if all skeletons are valid
    """
    arr = np.array([
        is_valid_skeleton(skeleton) for skeleton in torch.unbind(skeletons, dim=0)
    ])
    return np.all(arr)

def get_valid_skeletons(skeletons:torch.Tensor) -> list:
    """Return valid skeletons from skeletons list 

    Args:
        skeletons (torch.Tensor): shape = (N, 17, 2)

    Returns:
        list: list of valid skeletons shae = (N, 17, 2)
    """
    arr = np.array([
        is_valid_skeleton(skeleton) for skeleton in torch.unbind(skeletons, dim=0)
    ])
    return skeletons[arr]

def get_bbox_area(xmin, ymin, xmax, ymax) -> float:
    return (xmax-xmin)*(ymax-ymin)
            
def get_bbox(points:torch.Tensor) -> torch.Tensor:
    """Get bounding box of the skeleton

    Args:
        points (torch.Tensor): shape = (17, 2)

    Returns:
        torch.Tensor: return bbox
    """
    xmin, ymin = points.min(axis=0).values
    xmax, ymax = points.max(axis=0).values
    return torch.Tensor([xmin, ymin, xmax, ymax])

def get_mid_skeleton(skeletons:torch.Tensor, frame_orientation: int, reference_line: int) -> torch.Tensor:
    joint_idx = 0
    if frame_orientation == FrameDataConst.HORIZONTAL:
        joints = skeletons[:, joint_idx, 1]
    elif frame_orientation == FrameDataConst.VERTICAL:
        joints = skeletons[:, joint_idx, 0] 

    # Compute distances to the target line
    distances = torch.abs(joints - reference_line)

    # Find the closest skeleton
    closest_index = distances.argmin()

    return skeletons[closest_index].unsqueeze(0) 
