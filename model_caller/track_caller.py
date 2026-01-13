import numpy as np 
import logging

class TrackCaller:
    def __init__(self, window) -> None:
        self.window = window
        pass
    
class TrackCallerBbox(TrackCaller):
    def __init__(self, window):
        super().__init__(window)


    def get_iou(self, box1, box2):
        """
        box = (x, y, w, h) with x,y = top-left, w,h >= 0
        returns IoU in [0,1]
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # convert to (x1,y1,x2,y2)
        ax1, ay1 = x1, y1
        ax2, ay2 = x1 + max(0.0, w1), y1 + max(0.0, h1)
        bx1, by1 = x2, y2
        bx2, by2 = x2 + max(0.0, w2), y2 + max(0.0, h2)

        # intersection
        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h

        # areas & union
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0  

    def reassign_swimmer_id(self, valid_bboxes, valid_swimmer_ids, previous_bboxes_list, previous_ids_list):
        """_summary_

        Args:
            valid_bboxes (_type_): list of bboxes in a frame
            valid_swimmer_ids (_type_): list of swimmer ids in a frame
            previous_bboxes_list (_type_): list of bboxes in many frames
            previous_ids_list (_type_): list of swimmer ids in many frames
        Return: new list of swimmer ids
        """
        if len(previous_bboxes_list) == 0:
            return valid_swimmer_ids
        new_id_list = []
        previous_bboxes_list = previous_bboxes_list[-self.window:]
        previous_ids_list = previous_ids_list[-self.window:]
        for i in range(len(valid_bboxes)):
            cur_bbox = valid_bboxes[i] # 1 bbox
            cur_id = valid_swimmer_ids[i] # 1 id
            assigned_id = -1
            for j in range(len(previous_bboxes_list)-1,-1,-1): # run backwards
                previous_bboxes = previous_bboxes_list[j] # multiple bboxes in the previous frame
                previous_ids = previous_ids_list[j] # multiple ids in the previous frame
                iou_list = [self.get_iou(cur_bbox, previous_bbox) for previous_bbox in previous_bboxes]
                logging.info(f'iou_list = {iou_list}')
                if len(iou_list) == 0: # no bboxes detected
                    continue # go backwards to find
                argmax_idx = np.argmax(iou_list)
                # iou_list has maximum of 0 --> go backwards to find
                if iou_list[argmax_idx] == 0:
                    continue
                # iou_list has maximum of > 0.5 
                if iou_list[argmax_idx] > 0.5:
                    assigned_id = previous_ids[argmax_idx] # this swimmer id in the previous frame 
                    break
            if assigned_id == -1: # go backwards and can't find --> assign a new number
                if len(new_id_list) == 0:
                    assigned_id = max(valid_swimmer_ids) + 1
                else:
                    assigned_id = max(max(new_id_list),max(valid_swimmer_ids)) + 1
            new_id_list.append(assigned_id)
            logging.info(f'old list = {valid_swimmer_ids}, new_list = {new_id_list}')
        return np.array(new_id_list)


class TrackCallerSkeleton(TrackCaller):
    def __init__(self, window):
        super().__init__(window)

    def distance_x_y(self, p1, p2):
        return (abs(p1[0]-p2[0]), abs(p1[1]-p2[1])) # x, y

    def reassign_swimmer_id(self, valid_skeletons, valid_swimmer_ids, previous_skeletons_list, previous_ids_list):
        """_summary_

        Args:
            valid_skeletons (_type_): skeletons in one frame
            valid_swimmer_ids (_type_): swimmer ids in one frame
            previous_skeletons_list (_type_): list of skeletons in multiple frames
            previous_ids_list (_type_): list of ids in multiple frames
        """
        if len(previous_skeletons_list) == 0:
            return valid_swimmer_ids
        new_id_list = []
        previous_skeletons_list = previous_skeletons_list[-self.window:]
        previous_ids_list = previous_ids_list[-self.window:]
        for i in range(len(valid_skeletons)):
            cur_skeleton = valid_skeletons[i] # 1 skeleton
            # Keep tensor on device, only use .item() for scalar values
            head_position = cur_skeleton[0] # x and y keypoint
            threshold = max(50, abs(cur_skeleton[5,1].item() - cur_skeleton[6,1].item())) # shoulder in y axis
            cur_id = valid_swimmer_ids[i] # 1 id
            assigned_id = -1
            for j in range(len(previous_skeletons_list)-1,-1,-1): # run backwards
                previous_skeletons = previous_skeletons_list[j] # multiple skeletons in the previous frame
                previous_ids = previous_ids_list[j] # multiple ids in the previous frame
                distance_list = [self.distance_x_y(head_position, previous_skeleton[0]) for previous_skeleton in previous_skeletons]
                if len(distance_list) == 0: # no skeletons detected
                    continue # go backwards to find
                argmin_idx = min(range(len(distance_list)), key=lambda k: distance_list[k][1])
                if distance_list[argmin_idx][1] > threshold:
                    continue
                if distance_list[argmin_idx][1] <= threshold:
                    if argmin_idx < len(previous_ids):
                        assigned_id = previous_ids[argmin_idx] # this swimmer id in the previous frame 
                        break
            # if assigned_id == -1: # go backwards and can't find --> assign a new number
                # if len(new_id_list) == 0:
                    # assigned_id = max(valid_swimmer_ids) + 1
                # else:
                    # assigned_id = max(max(new_id_list),max(valid_swimmer_ids)) + 1
            new_id_list.append(assigned_id)
        max_id = max(new_id_list)
        # print(f'new_id_list={new_id_list}')
        for i in range(len(new_id_list)):
            # print(i, new_id_list[i])
            if new_id_list[i] == -1:
                new_id_list[i] = max_id + 1
                max_id += 1
            # print(f'old list = {valid_swimmer_ids}, new_list = {new_id_list}')
        return np.array(new_id_list)
