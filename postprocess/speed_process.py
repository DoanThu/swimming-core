import numpy as np 
from collections import OrderedDict
from postprocess.data import FrameData, FrameDataConst
import cv2
from utils.color_utils import filter_red
from scipy import signal
import scipy

class SpeedProcess:
    def __init__(self, fps):
        self.current_speed = FrameDataConst.UNKNOWN
        self.pct_change = FrameDataConst.UNKNOWN
        self.fps = fps
        self.frame_window = self.fps
        self.red_marker = False # to check if the head is at the red marker
        
        
    def calculate_speed(self, frame:np.ndarray, frame_data:FrameData, 
                        anchor_list:OrderedDict, 
                        lanes_segmentation:list):
        """ One frame might have multiple lane dividers. Each lane divider will return its length. The final distance will be the mode of all the dividers. 

        Args:
            frame (np.ndarray): 2D image
            frame_data (FrameData): metadata of the current frame
            anchor_list (OrderedDict): list of anchors, keys are time, values are coordinates    
            lanes_segmentation (list): segmentation of lane dividers detected
        """
        
        if len(anchor_list) < 2: return
        times = list(anchor_list.keys()) # times are already sorted

        # get begin and end time to calculate speed
        current_anchor = anchor_list[times[-1]][0]
        for i in range(len(times), -1, -1):
            if times[-1]-times[i-1] > self.frame_window: break
        past_anchor = anchor_list[times[i]][0]
        if times[-1] == times[i]: return 

        
        all_markers_count = [] # place to store markers count of all lane dividers
        if frame_data.frame_orientation == FrameDataConst.VERTICAL: # vertical, anchors at the same time have the same y
            y0, y1 = int(past_anchor[1]), int(current_anchor[1]) # get y coord
            if y1 < y0: y0, y1 = y1, y0 # make sure y1 >= y0
            for ilane, lane_segmentation in enumerate(lanes_segmentation):
                if len(lane_segmentation) == 0: continue
                x, y, w, h = lane_segmentation
                xmin, ymin, xmax, ymax = x, 0, x+w, frame.shape[0]
                
                if ymin < y0 and ymax > y1:
                    xmin, xmax = int(xmin), int(xmax)
                    if xmin - xmax == 0: continue
                    if y1 - y0 == 0:
                        all_markers_count.append(1)
                        continue
                    lane_image = frame[y0:y1,xmin:xmax,:]
                    # for debugging
                    cv2.imwrite(f'lanes/latest.jpg',lane_image)
                    
                    count_marker_ = y1-y0
                    self.red_marker = filter_red(lane_image).shape[0]>1
                    all_markers_count.append(count_marker_)

            
        else: # horizontal, anchors at the same time have the same x
            x0, x1 = int(past_anchor[0]), int(current_anchor[0]) # get x coord
            if x1 < x0: x0, x1 = x1, x0 # make sure x1 >= x0
            for ilane, lane_segmentation in enumerate(lanes_segmentation):
                if len(lane_segmentation) == 0: continue
                x, y, w, h = lane_segmentation
                xmin, ymin, xmax, ymax = 0, y, frame.shape[1], y+h
                if xmin < x0 and xmax > x1:
                    ymin, ymax = int(ymin), int(ymax)
                    if ymin - ymax == 0: continue
                    if x1 - x0 == 0:
                        all_markers_count.append(1)
                        continue
                    lane_image = frame[ymin:ymax,x0:x1,:]
                    cv2.imwrite('lanes/latest.jpg', lane_image)
                    
                    count_marker_ = x1-x0
                    self.red_marker = filter_red(lane_image).shape[0]>1
                    all_markers_count.append(count_marker_)

        
        all_markers_count = np.array(all_markers_count)
        if len(all_markers_count) == 0: return

        current_speed = np.mean(all_markers_count) / (times[-1]-times[i]) * self.fps
        if self.current_speed == FrameDataConst.UNKNOWN:
            self.current_speed = current_speed
        else:
            self.pct_change = (current_speed-self.current_speed)/self.current_speed
            self.current_speed = current_speed

