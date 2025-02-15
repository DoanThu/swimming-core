from ultralytics import YOLO
import torch
import numpy as np 
import cv2

class SegCaller:
    def __init__(self) -> None:
        pass
    
    def get_lane_dividers(self, image: np.ndarray, **kwargs) -> torch.Tensor:
        pass
    
class SegCallerYOLO(SegCaller):
    def __init__(self, model_path:str) -> None:
        super().__init__()
        self.model = YOLO(model_path)
        
    def get_lane_dividers(self, image: np.ndarray, **kwargs) -> list:
        """ Get lane divider segments

        Args:
            image (np.ndarray): 2D image

        Returns:
            list: list of divider segments. A list is returned because the segments are inhomogeneous in shape
        """
        results = self.model(image, **kwargs)
        masks = results[0].masks
        if masks == None: return []
        lane_divider_masks = [mask for i, mask in enumerate(results[0].masks.xy) if int(results[0].boxes.cls[i]) == 0]
        lane_divider_bboxes = [cv2.boundingRect(np.array(segment)) for segment in lane_divider_masks]
        return lane_divider_bboxes
        
        