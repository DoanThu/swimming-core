from ultralytics import YOLO
import torch

class FaceCaller:
    def __init__(self) -> None:
        pass
    
    def get_face(self, image, **kwargs):
        pass

class FaceCallerYOLO(FaceCaller):
    def __init__(self, model_path:str) -> None:
        super().__init__()
        self.model = YOLO(model_path)
        
    def get_face(self, image, **kwargs) -> torch.Tensor:
        """ Return bounding boxes of detected faces

        Args:
            image: 2-d image

        Returns:
            torch.Tensor: bounding boxes in xyxy format, shape = (N, 4)
        """
        results = self.model.predict(image, **kwargs)
        return results[0].boxes.xyxy
        
        