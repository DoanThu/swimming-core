import torchvision.transforms.functional as F
from torchvision.models.optical_flow import Raft_Small_Weights
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_small
from torchvision.models.optical_flow import raft_large

class OpticalFlowCaller:
    def __init__(self):
        pass

    def get_optical_flow(self, img1_batch, img2_batch):
        pass

class RAFTCaller(OpticalFlowCaller):
    def __init__(self, model_weight, device):
        super().__init__()
        self.device = device
        self.model_weight = model_weight
        if model_weight == 'small':
            self.model_weight = Raft_Small_Weights.DEFAULT
        elif model_weight == 'large':
            self.model_weight = Raft_Large_Weights.DEFAULT

        self.transforms = self.model_weight.transforms()
            
    def preprocess(self, img1_batch, img2_batch):
        img1_batch = F.resize(img1_batch, size=[520, 960], antialias=False)
        img2_batch = F.resize(img2_batch, size=[520, 960], antialias=False)
        return self.transforms(img1_batch, img2_batch)

    def get_optical_flow(self, img1_batch, img2_batch):
        img1_batch, img2_batch = self.preprocess(img1_batch, img2_batch)
        if self.model_weight == 'small':
            model = raft_small(weights=Raft_Large_Weights.DEFAULT, progress=False).to(self.device)
        elif self.model_weight == 'large':
            model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(self.device)
        model = model.eval()
        list_of_flows = model(img1_batch.to(self.device), img2_batch.to(self.device))

        print(f"type = {type(list_of_flows)}")
        print(f"length = {len(list_of_flows)} = number of iterations of the model")