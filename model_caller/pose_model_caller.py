from ultralytics import YOLO
import torch
import threading
import queue
import numpy as np 
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


class PoseCaller:
    def __init__(self) -> None:
        pass
    
    def get_keypoints(self, image, **kwargs):
        pass 
    
        
class PoseCallerYOLO(PoseCaller):
    def __init__(self, model_path:str) -> None:
        super().__init__()
        self.model = YOLO(model_path)
        
    def get_keypoints(self, image, **kwargs) -> torch.Tensor:
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
        if keypoints is None:
            return torch.zeros((0, 17, 2))
        return keypoints.xy
    
    def track_keypoints(self, image, **kwargs) -> torch.Tensor:
        """ Return keypoints from the model with tracking

        Args:
            image (_type_, optional): 2-D image 
            frame_idx (_type_, optional): None
            kargs: args for YOLOv8 prediction. Refer to this link: https://docs.ultralytics.com/modes/track/#persisting-tracks-loop
        Returns:
            dict: include keypoints, bboxes and track ids
        """
        results = self.model.track(image, persist=True, **kwargs, tracker='config/botsort.yaml')
        return results[0]
    
# This class is used to create a persistent thread for pose estimation
class PoseWorker(threading.Thread):
    """
    Persistent thread that owns the pose model and its CUDA context.
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
            # Move to CUDA once, fuse, warmup IN THIS THREAD
            self.model = PoseCallerYOLO(self.model_path)
            if hasattr(self.model, "model"):
                # self.model.model.to('cuda')
                try:
                    self.model.model.fuse()
                except Exception:
                    pass
            dummy = np.zeros((RESOLUTION[0], RESOLUTION[1], 3), np.uint8)
            with torch.inference_mode():
                for _ in range(3):
                    fn_gpu = getattr(self.model, "get_keypoints", None)
                    if callable(fn_gpu):
                        fn_gpu(dummy, **self.kwargs)
                    else:
                        self.model.get_keypoints(dummy, **self.kwargs)
            torch.cuda.synchronize()

            while not self.stop_flag.is_set():
                item = self.in_q.get()
                if item is None:
                    break
                frame, frame_idx = item
                with torch.inference_mode():
                    t0 = time.perf_counter()
                    fn_gpu = getattr(self.model, "get_keypoints", None)
                    if callable(fn_gpu):
                        res = fn_gpu(frame, **self.kwargs)   # should return CUDA tensors (no sync)
                    else:
                        res = self.model.get_keypoints(frame, **self.kwargs)  # may sync
                    t1 = time.perf_counter()
                # move any tensors in the result to CPU to avoid main-thread GPU sync
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
