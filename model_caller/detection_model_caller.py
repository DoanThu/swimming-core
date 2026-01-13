from ultralytics import YOLO
import torch
import numpy as np 

class DetectionCaller:
    def __init__(self) -> None:
        pass
    
    def detect_lane_dividers(self, image: np.ndarray, **kwargs) -> torch.Tensor:
        pass
    
class DetectionCallerYOLO(DetectionCaller):
    def __init__(self, model_path:str) -> None:
        super().__init__()
        self.model = YOLO(model_path)
        
    def detect_lane_dividers(self, image: np.ndarray, **kwargs) -> list:
        """ Detect lane dividers

        Args:
            image (np.ndarray): 2D image

        Returns:
            list: list of dividers
        """
        results = self.model(image, **kwargs)
        bboxes = results[0].boxes
        if bboxes == None: return []
        lane_divider_bboxes = [bbox for i, bbox in enumerate(results[0].boxes.xywh) if int(results[0].boxes.cls[i]) == 0]
        
        # Convert bboxes to list of lists while staying on device until final return
        if not lane_divider_bboxes:
            return []
        # Stack bboxes and convert them all at once if needed, more efficient
        stacked_bboxes = torch.stack(lane_divider_bboxes)
        final_bboxes = stacked_bboxes.tolist()
        return final_bboxes
        
        