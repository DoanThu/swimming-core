import torch
import numpy as np

class TrackCaller:
    def __init__(self, window) -> None:
        self.window = window

class TrackCallerSkeleton(TrackCaller):
    def __init__(self, window):
        super().__init__(window)

    def distance_x_y(self, p1: torch.Tensor, p2: torch.Tensor) -> tuple:
        """Compute absolute x,y distances between points, preserving device"""
        diff = torch.abs(p1 - p2)
        return diff[0].item(), diff[1].item()  # Only transfer scalars

    def reassign_swimmer_id(self, valid_skeletons, valid_swimmer_ids, previous_skeletons_list, previous_ids_list):
        """Track swimmers by skeleton keypoints across frames.

        Args:
            valid_skeletons (torch.Tensor): Skeletons in current frame
            valid_swimmer_ids (torch.Tensor): Swimmer IDs in current frame  
            previous_skeletons_list (list): List of skeleton tensors from previous frames
            previous_ids_list (list): List of ID lists from previous frames
        """
        if len(previous_skeletons_list) == 0:
            return valid_swimmer_ids

        # Keep processing window-sized
        previous_skeletons_list = previous_skeletons_list[-self.window:]
        previous_ids_list = previous_ids_list[-self.window:]
        
        device = valid_skeletons.device
        new_id_list = []
        
        for i in range(len(valid_skeletons)):
            cur_skeleton = valid_skeletons[i]
            # Extract scalar values only when needed for comparisons
            head_position = cur_skeleton[0]  # Keep as tensor
            threshold = max(50, abs(float(cur_skeleton[5,1] - cur_skeleton[6,1])))
            assigned_id = -1
            
            # Process each historical frame
            for j in range(len(previous_skeletons_list)-1, -1, -1):
                prev_skels = previous_skeletons_list[j]
                if isinstance(prev_skels, list):
                    # Convert historical data to tensor if needed
                    try:
                        prev_skels = torch.tensor(prev_skels, device=device)
                    except:
                        continue
                
                prev_ids = previous_ids_list[j]
                if len(prev_skels) == 0:
                    continue
                
                # Vectorized distance calculation
                head_diffs = torch.abs(prev_skels[:,0] - head_position)
                y_distances = head_diffs[:,1]
                
                # Find best match
                if len(y_distances) > 0:
                    min_dist, min_idx = torch.min(y_distances, dim=0)
                    if min_dist <= threshold:
                        assigned_id = prev_ids[min_idx]
                        break
            
            new_id_list.append(assigned_id)
        
        # Process new IDs
        if new_id_list:
            max_id = max(new_id_list)
            next_id = max_id + 1
            # Assign new IDs to unmatched skeletons
            for i in range(len(new_id_list)):
                if new_id_list[i] == -1:
                    new_id_list[i] = next_id
                    next_id += 1
        
        # Return IDs in consistent format
        return torch.tensor(new_id_list, device=device)