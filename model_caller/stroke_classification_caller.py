import torch
import torch.nn as nn
import math
import random
import numpy as np
import torch
from torchvision import transforms

SEQUENCE_LENGTH = 90 # 30 is 1 second
N_FEATURES = 17 * 3  # Flattened shape (17,3) -> 51
HIDDEN_SIZE = 32
OUTPUT_SIZE = 10  # output classes 
N_LAYERS = 3
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3

class StrokeClassificationCaller():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Deterministic preprocessing—no RandomRotate!
        self.transformation = transforms.Compose([
            NormalizeByShoulders(),
            CenterOnNose(),
            KeypointsToTensor(),
        ])

        # Load trained weights (remove DataParallel prefix if needed)
        state_dict = torch.load("models/best_model_classification-5-classes-rnn.pth", map_location=self.device)
        from collections import OrderedDict
        new_state = OrderedDict()
        for k,v in state_dict.items():
            new_state[k.replace("module.","")] = v

        self.model = RNNModel(N_FEATURES, HIDDEN_SIZE, OUTPUT_SIZE, N_LAYERS)
        self.model.load_state_dict(new_state)
        self.model.to(self.device).eval()
        
    
    def inference(self, skeletons):
        # skeletons has shape N,17,2
        skeletons = np.array(skeletons)
        # convert to N,17,3
        ones = np.ones((skeletons.shape[0], skeletons.shape[1], 1))
        skeletons = np.concatenate([skeletons, ones], axis=-1)
        T = len(skeletons)
        preds = []
        
        # sliding window over frames
        for i in range(T - SEQUENCE_LENGTH + 1):
            seq = skeletons[i : i + SEQUENCE_LENGTH, :]  
            # apply transforms -> Tensor (SEQUENCE_LENGTH, N_FEATURES)
            seq_t = self.transformation(seq).view(1, SEQUENCE_LENGTH, N_FEATURES).float().to(self.device)
            with torch.no_grad():
                out  = self.model(seq_t)                   # (1, OUTPUT_SIZE)
                lbl  = torch.argmax(out, dim=1).cpu().item()
            preds.append(lbl)      # assign to last frame of window
        
        preds = self.get_stroke(preds)
        return preds
    
    def get_stroke(self,pred_array):
        from collections import Counter
        counter = Counter(pred_array)
        sorted_items = counter.most_common()  # returns list of (key, count)
        for key, count in sorted_items:
            if key in [3,4,5,6]:
                # print(f"Selected key: {key} (count: {count})")
                return key



class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.3)

    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.norm(out)
        out = self.relu(out) 
        out = self.dropout(out)
        out = self.fc(out)
        return out
    


class RandomRotateWithKeypoints:
    def __call__(self, list_kpts):
        N = list_kpts.shape[0]  # Number of samples
        rotated_keypoints = []
        angle = random.choice([0, 90, 180, 270])
        
        for i in range(N):
            kpts = list_kpts[i]
            kpts = kpts.reshape(-1, 3)

            rotated = []
            for x,y,v in kpts:
                x_rotated, y_rotated = self._rotate_point(x, y, angle)
                rotated.append((x_rotated, y_rotated, v))
            rotated = np.array(rotated).reshape(-1)
            rotated_keypoints.append(rotated)
        return np.array(rotated_keypoints)
        
        

    def _rotate_point(self, x, y, angle_deg, cx=0, cy=0):
        # Rotate (x, y) around (cx, cy) by angle degrees
        angle_rad = math.radians(angle_deg)
        cos = math.cos(angle_rad)
        sin = math.sin(angle_rad)

        # Translate point to origin
        x -= cx
        y -= cy

        # Rotate
        x_new = x * cos - y * sin
        y_new = x * sin + y * cos

        # Translate back
        x_new += cx
        y_new += cy

        return x_new, y_new
    
class KeypointsToTensor:
    def __call__(self, keypoints):
        """
        keypoints: numpy array or list of shape (51,)
        returns: torch tensor of shape (51,) — dtype float32
        """
        if isinstance(keypoints, torch.Tensor):
            return keypoints.float()
        return torch.tensor(keypoints, dtype=torch.float32)
    
class CenterOnNose:
    def __call__(self, list_kpts):
        N = list_kpts.shape[0]  # Number of samples
        centralized_keypoints = []
        
        for i in range(N):
            kpts = list_kpts[i]
            kpts = self._center_on_nose(kpts)
            centralized_keypoints.append(kpts)
        return np.array(centralized_keypoints)  
    
    def _center_on_nose(self, kpts):
        kpts = kpts.reshape(-1, 3)
        nose_x = kpts[0][0]
        nose_y = kpts[0][1]
        for i in range(len(kpts)):
            x, y, v = kpts[i]
            x -= nose_x
            y -= nose_y
            kpts[i] = (x, y, v)
        return kpts.reshape(-1)
    
class NormalizeKeypoints:
    def __call__(self, list_kpts):
        N = list_kpts.shape[0]  # Number of samples
        normalized_keypoints = []
        
        for i in range(N):
            kpts = list_kpts[i]
            kpts = self._normalize(kpts)
            normalized_keypoints.append(kpts)
        return np.array(normalized_keypoints)  
    
    def _normalize(self, kpts):
        kpts = kpts.reshape(-1, 3)
        visible_keypoints = kpts[kpts[:, 2] > 0.5]
        if len(visible_keypoints) == 0:
            return kpts.reshape(-1)
        
        x_min, y_min = np.min(visible_keypoints[:, :2], axis=0)
        x_max, y_max = np.max(visible_keypoints[:, :2], axis=0)
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        if x_range == 0:
            x_range = 1
        if y_range == 0:
            y_range = 1
        
        for i in range(len(kpts)):
            x, y, v = kpts[i]
            x = (x - x_min) / x_range
            y = (y - y_min) / y_range
            kpts[i] = (x, y, v)
        
        return kpts.reshape(-1)
    
    
class NormalizeByShoulders:
    def __call__(self, list_kpts):
        N = list_kpts.shape[0]  # Number of samples
        normalized_keypoints = []
        
        for i in range(N):
            kpts = list_kpts[i]
            kpts = self._normalize_by_shoulder_width(kpts)
            normalized_keypoints.append(kpts)
        return np.array(normalized_keypoints)
    
    def _normalize_by_shoulder_width(self, keypoints):
        keypoints = keypoints.reshape(-1, 3) # keypoints: (17, 3) tensor (x, y, visibility)
        left = keypoints[5, :2]  # x, y
        right = keypoints[6, :2]
        
        shoulder_width = np.sqrt((left[0]-right[0])**2 + (right[1]-left[1])**2) + 1e-6

        # normalize all x, y by shoulder width
        keypoints[:, :2] = keypoints[:, :2] / shoulder_width
        return keypoints.reshape(-1)