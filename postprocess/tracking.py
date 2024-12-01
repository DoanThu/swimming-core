import torch 
import numpy as np

class SwimmerTracking:
    def __init__(self, coords: dict) -> None:
        """ Init coords for tracking

        Args:
            coords (dict): coordinates of swimmer position
        """
        self.x = coords['x']
        self.y = coords['y']
        self.skeleton = torch.Tensor(0,17,2)
        self.bbox = torch.Tensor(0,4)
        self.bbox_area = 0
    
    def update_position(self, new_coords: dict) -> None:
        """ Update tracking position

        Args:
            new_coords (dict): New coordinates
        """
        self.x = new_coords['x']
        self.y = new_coords['y']
        
        
    def get_closest_skeleton(self, skeletons: torch.Tensor) -> torch.Tensor:
        """ Get closest skeleton to the tracking point

        Args:
            skeletons (torch.Tensor): all keypoints detected in the image, shape = (N, 17, 2)

        Returns:
            torch.Tensor: assign the closest keypoint to self.skeleton, self.skeleton has shape = (1, 17, 2)
        """
        def get_bbox_area(xmin, ymin, xmax, ymax):
            return (xmax-xmin)*(ymax-ymin)
            
        def get_bbox(points:torch.Tensor):
            xmin, ymin = points.min(axis=0).values
            xmax, ymax = points.max(axis=0).values
            return torch.Tensor([xmin, ymin, xmax, ymax])
        
        def get_center(xmin, ymin, xmax, ymax):
            return (xmin+xmax)/2, (ymin+ymax)/2

        def dist(p1: torch.Tensor, p2: torch.Tensor) -> float:
            p1 = [_p.cpu().data.numpy() for _p in p1] 
            return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
        
        def get_threshold(points) -> float:
            xmin, ymin = points.min(axis=0).values
            xmax, ymax = points.max(axis=0).values
            return max(ymax-ymin,xmax-xmin)
        
        
        dist_thres = get_threshold(skeletons[0])
        bboxes = [get_bbox(skeleton) for skeleton in skeletons]
        centers = [get_center(*bbox) for bbox in bboxes]
        dist_arr = [dist(center, (self.x, self.y)) for center in centers]
        skeleton_idx = np.argmin(dist_arr)
        if dist_arr[skeleton_idx] > dist_thres:
            return 
        
        # Update new coordinates and update current skeleton
        self.update_position({'x':centers[skeleton_idx][0].cpu().data.numpy(),
                              'y':centers[skeleton_idx][1].cpu().data.numpy()})
        self.bbox = bboxes[skeleton_idx]
        self.bbox_area = get_bbox_area(*bboxes[skeleton_idx])
        self.skeleton =  skeletons[skeleton_idx][None, :, :]
