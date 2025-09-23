import torch
import numpy as np
from postprocess.single_data import FrameDataConst


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

def get_valid_skeletons_from_track(track_results:torch.Tensor) -> list:
    """Return valid skeletons ids from track_results 

    Args:
        track_results (dict): contains keypoints and ids 
    Returns:
        list: list of valid skeletons and their ids
    """
    keypoints = track_results.keypoints.xy
    ids = track_results.boxes.id
    arr = np.array([
        is_valid_skeleton(skeleton) for skeleton in torch.unbind(keypoints, dim=0)
    ])
    valid_skeletons = keypoints[arr]
    valid_ids = ids[arr]
    return valid_skeletons, valid_ids


def get_bbox_area(xmin, ymin, xmax, ymax) -> float:
    return (xmax-xmin)*(ymax-ymin)
            
def get_bbox(skeleton:torch.Tensor) -> torch.Tensor:
    """Get bounding box of the skeleton

    Args:
        points (torch.Tensor): shape = (17, 2)

    Returns:
        torch.Tensor: return bbox
    """
    if skeleton.shape[1] == 0: return (-1,)
    skeleton = skeleton.cpu().numpy()
    filtered_skeleton = skeleton[~((skeleton[:, 0] == 0) & (skeleton[:, 1] == 0))]
    xmin = int(np.min(filtered_skeleton[:, 0]))
    ymin = int(np.min(filtered_skeleton[:, 1]))
    xmax = int(np.max(filtered_skeleton[:, 0]))
    ymax = int(np.max(filtered_skeleton[:, 1]))

    # xmin, ymin = skeleton.min(axis=0).values
    # xmax, ymax = skeleton.max(axis=0).values
    # return torch.Tensor([xmin, ymin, xmax, ymax])
    return [xmin, ymin, xmax-xmin, ymax-ymin]

def get_mid_skeleton_old(skeletons:torch.Tensor, frame_orientation: int, reference_line: int) -> torch.Tensor:
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

def get_mid_skeleton(skeletons:torch.Tensor, frame_orientation: int, width: int, height: int):
    skeletons = skeletons.cpu().numpy()
    if skeletons.shape[1] == 0:
        return np.zeros((0,17,3))
    x_mid, y_mid = width/2, height/2
    
    if frame_orientation == FrameDataConst.HORIZONTAL:
        y_coords = skeletons[:, 0, 1]   # y 
        mask = abs(y_coords-y_mid)<=100
        skeletons = skeletons[mask]
        if len(skeletons) == 0:
            return np.zeros((0,17,3))
        x_coords = skeletons[:, 0, 0]    # x
        idx = np.argmin(abs(x_coords-x_mid))
        skeleton = skeletons[idx]
        
    elif frame_orientation == FrameDataConst.VERTICAL:
        x_coords = skeletons[:, 0, 0]   # x 
        mask = abs(x_coords-x_mid)<=100
        skeletons = skeletons[mask]
        if len(skeletons) == 0:
            return np.zeros((0,17,3)), (-1,)
        y_coords = skeletons[:, 0, 1]    # y
        idx = np.argmin(abs(y_coords-y_mid))
        skeleton = skeletons[idx]
    skeleton = torch.from_numpy(skeleton)
    return skeleton.unsqueeze(0)
    
    
    
    