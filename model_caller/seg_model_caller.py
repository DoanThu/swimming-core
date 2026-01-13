from ultralytics import YOLO
import torch
import numpy as np 
import cv2
import threading
import queue
import time
from config.general import RESOLUTION


def _to_cpu(obj):
    """Recursively move torch tensors in obj to CPU. Leaves other objects intact."""
    if torch.is_tensor(obj):
        try:
            return obj.detach().cpu()
        except Exception:
            return obj.cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_to_cpu(v) for v in obj]
        return type(obj)(converted)
    return obj

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
        start_time = time.perf_counter()
        results = self.model(image, **kwargs)
        masks = results[0].masks
        if masks == None: return []
        lane_divider_masks = [mask for i, mask in enumerate(results[0].masks.xy) if int(results[0].boxes.cls[i]) == 0]
        lane_divider_bboxes = [cv2.boundingRect(np.array(segment)) for segment in lane_divider_masks]
        lane_divider_bboxes = [bbox for bbox in lane_divider_bboxes if bbox[2] != 0 and bbox[3] != 0]
        return lane_divider_bboxes
    
# This class is used to create a persistent thread for segmentation        
class SegWorker(threading.Thread):
    """
    Persistent thread that owns the segmentation model and its CUDA context.
    Send (frame, frame_idx) via input_q; receive (frame_idx, result, (t0, t1)) via output_q.
    """
    def __init__(self, model_path: str, 
                 input_q: "queue.Queue", output_q: "queue.Queue",
                 **kwargs):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.in_q = input_q
        self.out_q = output_q
        self.stop_flag = threading.Event()
        self.model = None
        self.kwargs = kwargs


    def run(self):
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(1)
        try:
            self.model = SegCallerYOLO(self.model_path)
            if hasattr(self.model, "model"):
                self.model.model.to('cuda')
                try:
                    self.model.model.fuse()
                except Exception:
                    pass
            dummy = np.zeros((RESOLUTION[0], RESOLUTION[1], 3), np.uint8)
            with torch.inference_mode():
                for _ in range(3):
                    # fn_gpu = getattr(self.model, "get_lane_dividers", None)
                    # if callable(fn_gpu):
                        # fn_gpu(dummy, **self.kwargs)
                    # else:
                    self.model.get_lane_dividers(dummy, **self.kwargs)
            torch.cuda.synchronize()

            while not self.stop_flag.is_set():
                item = self.in_q.get()
                if item is None:
                    break
                frame, frame_idx = item
                with torch.inference_mode():
                    t0 = time.perf_counter()
                    res = self.model.get_lane_dividers(frame, **self.kwargs)  # may sync
                    t1 = time.perf_counter()
                # Convert any tensors to CPU to avoid main-thread GPU sync
                try:
                    res_cpu = _to_cpu(res)
                except Exception:
                    res_cpu = res
                self.out_q.put((frame_idx, res_cpu, (t0, t1)))
                self.in_q.task_done()
        finally:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

