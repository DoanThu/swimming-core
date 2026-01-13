import torch
import numpy as np
from postprocess.single_data import FrameDataConst


def is_valid_skeleton(skeleton: torch.Tensor) -> bool:
    """Return if the skeleton is valid.

    A valid skeleton is one that does not have all keypoints equal to (0,0).
    The implementation is device-agnostic and avoids creating new CUDA tensors per call.

    Args:
        skeleton (torch.Tensor): shape = (17,2)

    Returns:
        bool: True if the skeleton has at least one non-zero joint
    """
    # skeleton may be on CPU or CUDA; do the comparison in-place on the same device
    # A skeleton is invalid iff all joints are (0,0). We return True when it's valid.
    if not torch.is_tensor(skeleton):
        # Non-tensor input -> treat as invalid
        return False
    # (17,2) -> compare elementwise to zero, then check if every joint is zero
    all_zero_per_joint = torch.all(skeleton == 0, dim=1)  # (17,) bool tensor
    # skeleton is valid if NOT all joints are zero
    # .any() returns a tensor; use .item() to convert single boolean (cheap)
    return (not torch.all(all_zero_per_joint).item())


def is_valid_skeletons(skeletons: torch.Tensor) -> bool:
    """Return True if every skeleton in the batch is valid.

    Vectorized implementation to avoid Python loops and per-call device allocations.
    """
    if not torch.is_tensor(skeletons) or skeletons.numel() == 0:
        return False
    # skeletons: (N,17,2) -> per-joint all-zero mask: (N,17)
    all_zero_per_joint = torch.all(skeletons == 0, dim=2)
    # a skeleton is invalid if all its joints are zero; we want all skeletons to be valid
    all_skeletons_valid = torch.logical_not(torch.all(all_zero_per_joint, dim=1)).all()
    return bool(all_skeletons_valid)

def get_valid_skeletons(skeletons: torch.Tensor) -> torch.Tensor:
    """Return valid skeletons from a batch.

    Vectorized and device-agnostic: avoids per-item Python loops and extra CUDA ops.
    Args:
        skeletons (torch.Tensor): shape = (N, 17, 2)

    Returns:
        torch.Tensor: filtered skeletons (M,17,2) where M <= N
    """
    if not torch.is_tensor(skeletons) or skeletons.numel() == 0:
        # return an empty tensor with expected rank
        return skeletons.new_zeros((0, 17, 2))
    # all_zero_per_joint: (N,17) bool
    all_zero_per_joint = torch.all(skeletons == 0, dim=2)
    valid_mask = ~torch.all(all_zero_per_joint, dim=1)
    return skeletons[valid_mask]

def get_valid_skeletons_from_track(track_results: torch.Tensor) -> tuple:
    """Return valid skeletons and ids from track results.

    Uses vectorized logic similar to get_valid_skeletons.
    """
    keypoints = track_results.keypoints.xy
    ids = track_results.boxes.id
    if ids is None:
        return keypoints.new_zeros((0, 17, 2)), []
    if keypoints.numel() == 0:
        return keypoints.new_zeros((0, 17, 2)), []
    all_zero_per_joint = torch.all(keypoints == 0, dim=2)
    valid_mask = ~torch.all(all_zero_per_joint, dim=1)
    valid_skeletons = keypoints[valid_mask]
    valid_ids = ids[valid_mask]
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
    # Handle empty input
    if not torch.is_tensor(skeleton) or skeleton.numel() == 0:
        return (-1,)

    # If given a batch with shape (1,17,2), squeeze to (17,2)
    if skeleton.dim() == 3 and skeleton.shape[0] == 1:
        skeleton = skeleton.squeeze(0)

    # Ensure we have shape (17,2)
    if skeleton.dim() != 2 or skeleton.shape[1] < 2:
        return (-1,)

    # Create a boolean mask of valid points (not both 0)
    valid_mask = ~((skeleton[:, 0] == 0) & (skeleton[:, 1] == 0))
    if valid_mask.sum().item() == 0:
        return (-1,)

    valid_points = skeleton[valid_mask]

    # Use torch ops (device-agnostic) and convert scalars to Python ints (tiny transfer)
    xmin = int(torch.min(valid_points[:, 0]).item())
    ymin = int(torch.min(valid_points[:, 1]).item())
    xmax = int(torch.max(valid_points[:, 0]).item())
    ymax = int(torch.max(valid_points[:, 1]).item())

    return [xmin, ymin, xmax - xmin, ymax - ymin]

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
    # Keep operations on device until final conversion
    if skeletons.shape[1] == 0:
        return torch.zeros((0,17,3), device=skeletons.device)
    x_mid, y_mid = width/2, height/2
    
    if frame_orientation == FrameDataConst.HORIZONTAL:
        y_coords = skeletons[:, 0, 1]   # y 
        mask = torch.abs(y_coords-y_mid)<=100
        skeletons = skeletons[mask]
        if skeletons.shape[0] == 0:
            return torch.zeros((0,17,3), device=skeletons.device)
        x_coords = skeletons[:, 0, 0]    # x
        idx = torch.argmin(torch.abs(x_coords-x_mid))
        skeleton = skeletons[idx]
        
    elif frame_orientation == FrameDataConst.VERTICAL:
        x_coords = skeletons[:, 0, 0]   # x 
        mask = torch.abs(x_coords-x_mid)<=100
        skeletons = skeletons[mask]
        if skeletons.shape[0] == 0:
            return torch.zeros((0,17,3), device=skeletons.device)
        y_coords = skeletons[:, 0, 1]    # y
        idx = torch.argmin(torch.abs(y_coords-y_mid))
        skeleton = skeletons[idx]
    return skeleton.unsqueeze(0)
    
    
    
    