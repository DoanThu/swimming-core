# https://pytorch.org/vision/0.19/auto_examples/others/plot_optical_flow.html

import torchvision.transforms.functional as F
from torchvision.models.optical_flow import Raft_Small_Weights
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_small
from torchvision.models.optical_flow import raft_large
from config.general import RESOLUTION

class OpticalFlowCaller:
    def __init__(self):
        pass

    def get_optical_flow(self, img1_batch, img2_batch):
        pass

class RAFTCaller(OpticalFlowCaller):
    def __init__(self, model_type, device):
        super().__init__()
        self.device = device
        self.model_type = model_type
        if model_type == 'small':
            self.model_weight = Raft_Small_Weights.DEFAULT
        elif model_type == 'large':
            self.model_weight = Raft_Large_Weights.DEFAULT

        self.transforms = self.model_weight.transforms()
            
    def preprocess(self, img1_batch, img2_batch):
        img1_batch = F.resize(img1_batch, size=[RESOLUTION[0], RESOLUTION[1]], antialias=False)
        img2_batch = F.resize(img2_batch, size=[RESOLUTION[0], RESOLUTION[1]], antialias=False)
        return self.transforms(img1_batch, img2_batch)

    def get_optical_flow(self, img1_batch, img2_batch):
        # resize due to the requirement of the model
        img1_batch, img2_batch = self.preprocess(img1_batch, img2_batch)
        if self.model_type == 'small':
            model = raft_small(weights=self.model_weight, progress=False).to(self.device)
        elif self.model_type == 'large':
            model = raft_large(weights=self.model_weight, progress=False).to(self.device)
        model = model.eval()
        list_of_flows = model(img1_batch.to(self.device), img2_batch.to(self.device))
        predicted_flows = list_of_flows[-1]
        return predicted_flows
