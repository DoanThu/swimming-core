import numpy as np
from model_caller.seg_model_caller import SegCaller
from postprocess.data import FrameData, FrameDataConst
from collections import OrderedDict
import cv2
from model_caller.optical_flow_model_caller import OpticalFlowCaller
import torch


class AnchorProcess:
    def __init__(self, window: int = 60, method: str = 'LK', optical_flow_model: OpticalFlowCaller = None):
        self.prev_frame = np.array([])
        # dictionary of time: adjusted anchors' coordinates
        self.anchor_list = OrderedDict()
        self.random_anchor_list = OrderedDict() # generated random anchor points to minimize the effect of water flow
        self.window = window
        self.lk_params = {'winSize':(15, 15), 'maxLevel':2,
                           'criteria':(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}
        self.method = method
        self.optical_flow_model = optical_flow_model


    def segment_lane_dividers(self, frame: np.ndarray, frame_data: FrameData,
                              seg_caller: SegCaller, padding_coef:int=5, **seg_kwargs) -> list:
        """ Return a list of lane divider segments. The image passed to the segmentation model is centered at the head of the swimmer to make sure the head and the segments are aligned. 
        The segments' coordinates are converted into the original frame's coordinate in the end.

        Args:
            frame (np.ndarray): 2D image
            frame_data (FrameData): metadata of the current frame
            skeleton (torch.Tensor): tracked skeleton, shape = (1, 17, 2)
            seg_caller (SegCaller): caller to the segmentation model

        Returns:
            list: list of lane divider segments
        """

        # crop the image s.t. the head is in the center
        skeleton = frame_data.skeleton
        head_coord = skeleton[0][0].cpu().numpy()
        xmin, ymin, xmax, ymax = frame_data.bbox.cpu().numpy()
        # cropping window x1,y1,x2,y2
        if frame_data.frame_orientation == FrameDataConst.VERTICAL: # vertical frame
            padding = (ymax-ymin)*padding_coef
            x1, y1, x2, y2 = 0, max(
                0, head_coord[1]-padding), frame.shape[1], min(head_coord[1]+padding, frame.shape[0])
        else:  # horizontal frame and unknown
            padding = (xmax-xmin)*padding_coef
            x1, y1, x2, y2 = max(0, head_coord[0]-padding), 0, min(
                head_coord[0]+padding, frame.shape[1]), frame.shape[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cropped_frame = frame[y1:y2, x1:x2:]
        lane_dividers = seg_caller.get_lane_dividers(cropped_frame, **seg_kwargs)

        # convert lane dividers coords to original coord
        for i in range(len(lane_dividers)):
            lane_dividers[0][:, 0] += x1
            lane_dividers[0][:, 1] += y1

        return lane_dividers
    
    def get_updated_vectors(self, p0, frame):
        if self.method == 'LK':
            # reshape according to optical flow lib's requirement
            p0 = p0.reshape(p0.shape[0],1,p0.shape[1])
            p1, st, err = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                                    p0, None, **self.lk_params)
            good_new, good_old = [], []
            if p1 is not None:
                good_new = p1[st==1]
                good_old = p0[st==1]

            # Get updated position of anchor points
            updated_vectors = []
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                updated_vectors.append([a-c, b-d])
            return updated_vectors
        elif self.method == 'RAFT':
            # example of input
            # img1_batch = torch.stack([frames[100], frames[150]])
            # img2_batch = torch.stack([frames[101], frames[151]])

            def to_tensor_image(image):
                tensor_image = torch.from_numpy(image)
                tensor_image = tensor_image.permute(2, 0, 1)
                tensor_image = tensor_image.unsqueeze(0)
                return tensor_image[0]


            img1_batch = torch.stack([to_tensor_image(self.prev_frame)])
            img2_batch = torch.stack([to_tensor_image(frame)])
            predicted_flow = self.optical_flow_model.get_optical_flow(img1_batch, img2_batch)
            predicted_flow = predicted_flow.detach().cpu().numpy()  # (N, 2, H, W)

            updated_vectors = []
            for x_, y_ in p0:
                x_, y_ = int(x_), int(y_)
                x_vector = predicted_flow[0][0][x_][y_]
                y_vector = predicted_flow[0][1][x_][y_]
                updated_vectors.append([x_vector, y_vector])
            return updated_vectors


    def update_optical_flow_anchors(self, frame: np.ndarray):
        """ Update previous anchor points using optical flow. 
        Also remove updated points that are out-of-sight along the way.

        Args:
            frame (np.ndarray): 2D image from the current frame.
        """
        if self.prev_frame.shape[0] == 0: # first frame
            self.prev_frame = frame
            return
        if len(self.anchor_list) == 0:
            return
        
        keys_to_remove = []

        latest_point = list(self.random_anchor_list.keys())[0]
        p0 = self.random_anchor_list[latest_point]

        updated_vectors = self.get_updated_vectors(p0, frame)

        # Customize to update anchor points
        updated_vectors = np.array(updated_vectors)
        if len(updated_vectors) == 0: return
        filter_x = self.reject_outliers(updated_vectors[:,0])
        filter_y = self.reject_outliers(updated_vectors[:,1])
        updated_vector = np.array([np.mean(filter_x), np.mean(filter_y)])

        # Update anchor points
        for k, p0 in self.anchor_list.items():
            new_p = p0 + updated_vector
            adjusted_anchors = []
            for p in new_p:
                if np.any(p < 0) or np.isnan(p).any() or p[0] >= frame.shape[1] or p[1] >= frame.shape[0]:
                    continue
                adjusted_anchors.append(p)
            if len(adjusted_anchors) != 0:
                self.anchor_list[k] = np.array(adjusted_anchors, dtype=np.float32)
            else:
                keys_to_remove.append(k)

        # remove key whose values have length 0
        for k in keys_to_remove:
            self.anchor_list.pop(k)

        # Update random anchor points
        for k, p0 in self.random_anchor_list.items():
            new_p = p0 + updated_vector
            adjusted_anchors = []
            for p in new_p:
                if np.any(p < 0) or np.isnan(p).any() or p[0] >= frame.shape[1] or p[1] >= frame.shape[0]:
                    continue
                adjusted_anchors.append(p)
            if len(adjusted_anchors) != 0:
                self.random_anchor_list[k] = np.array(adjusted_anchors, dtype=np.float32)
            else:
                keys_to_remove.append(k)

        # remove key whose values have length 0
        for k in keys_to_remove:
            self.random_anchor_list.pop(k)



        self.prev_frame = frame
        

    
    def reject_outliers(self, data, m = 2.):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else np.zeros(len(d))
        return data[s<m]
            
            
    def update_anchor_points(self, frame_idx: int, frame: np.ndarray, frame_data: FrameData,
                             lane_dividers: list):
        """ First, update previous anchor points using optical flow.
        Then, add new points if there are new intersection points with the lane dividers.
        Keep the anchor points as they are, if lane divider list is empty or there are not new intersection points found.

        Args:
            frame_idx (int): index of the frame. Required to keep track of time.
            frame (np.ndarray): 2D image
            frame_data (FrameData): metadata of the current frame
            lane_dividers (list): list of lane dividers

        """
        
        # shape for anchors must be (N, 1, 2)
        # note the np.float32 too

        # Updated previous anchors 
        # And remove out-of-sight anchors in anchor list (negative coordinates)
        self.update_optical_flow_anchors(frame)
        

        # Remove anchors that are beyond time window
        # To keep the size of the list small
        keys_to_remove = [k for k in self.anchor_list.keys()
                          if k + self.window < frame_idx]
        for k in keys_to_remove:
            self.anchor_list.pop(k)
        
        keys_to_remove = [k for k in self.random_anchor_list.keys()
                          if k + self.window < frame_idx]
        for k in keys_to_remove:
            self.random_anchor_list.pop(k)
        

        if len(lane_dividers) != 0:
            # Add more anchors to anchor_list
            skeleton = frame_data.skeleton
            head_coord = skeleton[0][0].cpu().numpy()
            new_anchors = []
            if frame_data.frame_orientation == FrameDataConst.VERTICAL and frame_data.direction in [FrameDataConst.UP, FrameDataConst.DOWN]:  # vertical frame
                reference_y = head_coord[1]  # y coord
                for divider in lane_dividers:
                    x, y, w, h = divider
                    new_anchors.extend([[x, reference_y], [x+w, reference_y]])
                        
            elif frame_data.frame_orientation == FrameDataConst.HORIZONTAL and frame_data.direction in [FrameDataConst.LEFT, FrameDataConst.RIGHT]:  # horizontal frame
                reference_x = head_coord[0]  # x coord
                for divider in lane_dividers:
                    x, y, w, h = divider
                    new_anchors.extend([[reference_x, y], [reference_x, y+h]])

            if len(new_anchors) == 0:
                return
            self.anchor_list[frame_idx] = np.array(new_anchors, np.float32)

    def add_random_anchor_points(self, frame_idx, random_x, random_y):
        """ Add random anchor points to minimize the effect of water flow.

        Args:
            random_x (np.ndarray): x coordinates of random anchor points
            random_y (np.ndarray): y coordinates of random anchor points
        """
        temp = np.array([[x, y] for x, y in zip(random_x, random_y)], np.float32)
        if frame_idx not in self.random_anchor_list:
            self.random_anchor_list[frame_idx] = temp
        else:
            self.random_anchor_list[frame_idx] = np.concatenate((self.random_anchor_list[frame_idx], temp), axis=0)