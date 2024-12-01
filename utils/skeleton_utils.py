import torch
import numpy as np

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

def get_valid_skeletons(skeletons:torch.Tensor) -> bool:
    arr = np.array([
        is_valid_skeleton(skeleton) for skeleton in torch.unbind(skeletons, dim=0)
    ])
    return skeletons[arr]