from ultralytics import YOLO
import torch


class PoseCaller:
    def __init__(self) -> None:
        pass
    
    def get_keypoint(self, image, **kwargs):
        pass 
    
        
class PoseCallerYOLO(PoseCaller):
    def __init__(self, model_path:str) -> None:
        super().__init__()
        self.model = YOLO(model_path)
        
    def get_keypoint(self, image, **kwargs) -> torch.Tensor:
        """ Return keypoints from the model

        Args:
            image (_type_, optional): 2-D image 
            frame_idx (_type_, optional): None
            kargs: args for YOLOv8 prediction. Refer to this link: https://docs.ultralytics.com/modes/predict/#inference-arguments

        Returns:
            dict: 17 keypoints (x,y) of n people, shape = (n,17,2)
        """
        results = self.model(image, **kwargs)
        keypoints = results[0].keypoints
        return keypoints.xy