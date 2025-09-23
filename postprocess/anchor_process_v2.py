import numpy as np
from postprocess.multiple_data import FrameMultipleData
from postprocess.const import FrameDataConst
from collections import OrderedDict
import cv2

class AnchorPoint:
    def __init__(self, coord:np.ndarray, 
                 swimmer_id: int = -1):
        self.coord = coord  # np.ndarray of shape (2,)
        self.swimmer_id = swimmer_id  # -1 for random points, otherwise swimmer id

    def __str__(self):
        return f"AnchorPoint(coord={self.coord}, swimmer_id={self.swimmer_id})"
    
    def __repr__(self):
        return self.__str__()


class AnchorProcess:
    def __init__(self, window: int = 60, method: str = 'LK'):
        # dictionary of time: adjusted anchors' coordinates
        self.anchor_list = OrderedDict()
        self.random_anchor_list = OrderedDict() # generated random anchor points to minimize the effect of water flow
        self.window = window
        self.lk_params = {'winSize':(15, 15), 'maxLevel':2,
                           'criteria':(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}
        self.method = method


    def get_updated_vectors(self, p0, frame, prev_frame):
        if self.method == 'LK':
            # reshape according to optical flow lib's requirement
            p0 = p0.reshape(p0.shape[0],1,p0.shape[1])
            p1, st, err = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
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



    def update_optical_flow_anchors(self, frame: np.ndarray, prev_frame: np.ndarray):
        """ Update previous anchor points using optical flow. 
        Also remove updated points that are out-of-sight along the way.

        Args:
            frame (np.ndarray): 2D image from the current frame.
        """
        if len(self.anchor_list) == 0:
            return
        

        latest_point_idx = list(self.random_anchor_list.keys())[0] # index of the latest AnchorPoint
        p0 = self.random_anchor_list[latest_point_idx]
        p0 = np.array([point.coord for point in p0], dtype=np.float32)

        # updated vectors are based on random anchor points
        updated_vectors = self.get_updated_vectors(p0, frame, prev_frame)

        # Use mean of updated vectors to update all anchor points
        updated_vectors = np.array(updated_vectors)
        if len(updated_vectors) == 0: 
            return
        filter_x = self.reject_outliers(updated_vectors[:,0])
        filter_y = self.reject_outliers(updated_vectors[:,1])
        updated_vector = np.array([np.mean(filter_x), np.mean(filter_y)])


        keys_to_remove = []
        # Update anchor points
        for k, p0 in self.anchor_list.items(): # p0 is a list of AnchorPoint
            new_p = [ap.coord + updated_vector for ap in p0] 
            swimmer_ids = [ap.swimmer_id for ap in p0]
            adjusted_anchors = []
            for p, swimmer_id in zip(new_p, swimmer_ids):
                if np.any(p < 0) or np.isnan(p).any() or p[0] >= frame.shape[1] or p[1] >= frame.shape[0]:
                    continue
                adjusted_anchors.append((p, swimmer_id))
            if len(adjusted_anchors) != 0:
                self.anchor_list[k] = np.array([AnchorPoint(coord=aa[0], swimmer_id=aa[1]) for aa in adjusted_anchors])
            else:
                keys_to_remove.append(k)

        # remove key whose values have length 0
        for k in keys_to_remove:
            self.anchor_list.pop(k)

        keys_to_remove = []
        # Update random anchor points
        for k, p0 in self.random_anchor_list.items():
            new_p = [ap.coord + updated_vector for ap in p0] 
            swimmer_ids = [ap.swimmer_id for ap in p0]
            adjusted_anchors = []
            for p, swimmer_id in zip(new_p, swimmer_ids):
                if np.any(p < 0) or np.isnan(p).any() or p[0] >= frame.shape[1] or p[1] >= frame.shape[0]:
                    continue
                adjusted_anchors.append((p, swimmer_id))
            if len(adjusted_anchors) != 0:
                self.random_anchor_list[k] = np.array([AnchorPoint(coord=aa[0], swimmer_id=aa[1]) for aa in adjusted_anchors])
            else:
                keys_to_remove.append(k)

        # remove key whose values have length 0
        for k in keys_to_remove:
            self.random_anchor_list.pop(k)

    
    def reject_outliers(self, data, m = 2.):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else np.zeros(len(d))
        return data[s<m]
            
            
    def update_anchor_points(self, frame_idx: int, 
                             frame: np.ndarray, previous_frame: np.ndarray,
                             frame_multi_data: FrameMultipleData,
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
        # And remove out-of-sight anchors in anchor list (negative coordinates or out of frame)
        self.update_optical_flow_anchors(frame, previous_frame)
        

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
            # add head projection points on lane dividers as new anchors
            for i in range(len(frame_multi_data.swimmer_id_list)):
                skeleton = frame_multi_data.skeleton_list[i]
                head_coord = skeleton[0].cpu().numpy()
                new_anchors = []
                if frame_multi_data.frame_orientation == FrameDataConst.VERTICAL and frame_multi_data.direction_list[i] in [FrameDataConst.UP, FrameDataConst.DOWN]:  # vertical frame
                    reference_y = head_coord[1]  # y coord
                    for divider in lane_dividers:
                        x, y, w, h = divider
                        new_anchors.extend([[x, reference_y, frame_multi_data.swimmer_id_list[i]],
                                             [x+w, reference_y, frame_multi_data.swimmer_id_list[i]]])
                        
                elif frame_multi_data.frame_orientation == FrameDataConst.HORIZONTAL and frame_multi_data.direction_list[i] in [FrameDataConst.LEFT, FrameDataConst.RIGHT]:  # horizontal frame
                    reference_x = head_coord[0]  # x coord
                    for divider in lane_dividers:
                        x, y, w, h = divider
                        new_anchors.extend([[reference_x, y, frame_multi_data.swimmer_id_list[i]],
                                             [reference_x, y+h, frame_multi_data.swimmer_id_list[i]]])

                if len(new_anchors) == 0: continue
                if frame_idx not in self.anchor_list:
                    self.anchor_list[frame_idx] = np.array([AnchorPoint(np.array([na[0], na[1]], dtype=np.float32), swimmer_id=na[2]) for na in new_anchors])
                else:
                    self.anchor_list[frame_idx] = np.concatenate((self.anchor_list[frame_idx], 
                                                                 np.array([AnchorPoint(np.array([na[0], na[1]], dtype=np.float32), swimmer_id=na[2]) for na in new_anchors])), axis=0)
    
    def add_random_anchor_points(self, frame_idx, random_x, random_y):
        """ Add random anchor points to minimize the effect of water flow.

        Args:
            random_x (np.ndarray): x coordinates of random anchor points
            random_y (np.ndarray): y coordinates of random anchor points
        """
        temp = np.array([AnchorPoint(np.array([x, y], dtype=float), swimmer_id=-1) for x, y in zip(random_x, random_y)])
        if frame_idx not in self.random_anchor_list:
            self.random_anchor_list[frame_idx] = temp
        else:
            self.random_anchor_list[frame_idx] = np.concatenate((self.random_anchor_list[frame_idx], temp), axis=0)