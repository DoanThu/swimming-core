import numpy as np 
from collections import OrderedDict
from postprocess.data import FrameData, FrameDataConst
import cv2
from utils.color_utils import filter_red, filter_blue, filter_yellow
from scipy import signal
import scipy

class SpeedProcess:
    def __init__(self, frame_window:int=60, marker_size:float=1, fps:int=60):
        self.frame_window = frame_window
        self.current_speed = FrameDataConst.UNKNOWN
        self.pct_change = FrameDataConst.UNKNOWN
        self.marker_size = marker_size # size of the marker in reality im meter. Default is 1.
        self.fps = fps
        self.marker_size_pixel = -1
        self.red_marker = False # to check if the head is at the red marker
        
        
    def calculate_speed(self, frame:np.ndarray, frame_data:FrameData, 
                        anchor_list:OrderedDict, 
                        lanes_segmentation:list,
                        unit_size:int):
        """ One frame might have multiple lane dividers. Each lane divider is passed into method self.sum_markers to get the number of marker on each divider. The final distance will be the mode of all the dividers. 

        Args:
            frame (np.ndarray): 2D image
            frame_data (FrameData): metadata of the current frame
            anchor_list (OrderedDict): list of anchors, keys are time, values are coordinates    
            lanes_segmentation (list): segmentation of lane dividers detected
        """
        self.marker_size_pixel = unit_size
        
        def get_bbox(points:np.ndarray):
            xmin, xmax = np.min(points[:,0]), np.max(points[:,0])
            ymin, ymax = np.min(points[:,1]), np.max(points[:,1])
            return xmin, ymin, xmax, ymax
        
        if len(anchor_list) < 2: return
        times = list(anchor_list.keys()) # times are already sorted
        
        # get begin and end time to calculate speed
        current_anchor = anchor_list[times[-1]][0]
        for i in range(len(times), -1, -1):
            if times[-1]-times[i-1] > self.frame_window: break
        past_anchor = anchor_list[times[i]][0]
        if times[-1]  == times[i]: return 
        
        all_markers_count = [] # place to store markers count of all lane dividers
        if frame_data.frame_orientation == FrameDataConst.VERTICAL: # vertical, anchors at the same time have the same y
            y0, y1 = int(past_anchor[1]), int(current_anchor[1]) # get y coord
            if y1 < y0: y0, y1 = y1, y0 # make sure y1 >= y0
            # if abs(y0-y1) < 10: return # filter if the difference is too little
            for ilane, lane_segmentation in enumerate(lanes_segmentation):
                if len(lane_segmentation) == 0: continue
                xmin, ymin, xmax, ymax = get_bbox(lane_segmentation)
                if ymin < y0 and ymax > y1:
                    xmin, xmax = int(xmin), int(xmax)
                    if xmin - xmax == 0: continue
                    if y1 - y0 == 0:
                        all_markers_count.append(1)
                        continue
                    # print(f'vertical: y0,y1 = {y0},{y1}')
                    lane_image = frame[y0:y1,xmin:xmax,:]
                    # for debugging
                    # cv2.imwrite(f'lanes/latest_{ilane}.jpg',lane_image)
                    cv2.imwrite(f'lanes/latest.jpg',lane_image)
                    
                    # count_marker_ = self.sum_markers(lane_image)
                    count_marker_ = y1-y0
                    self.red_marker = filter_red(lane_image).shape[0]>1
                    # if filter_red(lane_image).shape[0]>1:
                        # self.red_marker = True
                    # else:
                        # self.red_marker = False
                    if count_marker_ != -1: # == -1  when it is likely not a lane divider 
                        all_markers_count.append(count_marker_)
            
        else: # horizontal, anchors at the same time have the same x
            x0, x1 = int(past_anchor[0]), int(current_anchor[0]) # get x coord
            if x1 < x0: x0, x1 = x1, x0 # make sure x1 >= x0
            # if abs(x0-x1) < 10: return # filter if the difference is too little
            for ilane, lane_segmentation in enumerate(lanes_segmentation):
                if len(lane_segmentation) == 0: continue
                xmin, ymin, xmax, ymax = get_bbox(lane_segmentation)
                if xmin < x0 and xmax > x1:
                    ymin, ymax = int(ymin), int(ymax)
                    # print(f'horizontal: x0,x1 = {x0},{x1}')
                    # print(f'horizontal: ymin,ymax = {ymin},{ymax}')
                    if ymin - ymax == 0: continue
                    if x1 - x0 == 0:
                        all_markers_count.append(1)
                        continue
                    lane_image = frame[ymin:ymax,x0:x1,:]

                    # cv2.countNonZero(mask) > 0
                    # print(lane_image.shape)
                    # for debugging
                    # cv2.imwrite(f'lanes/latest_{ilane}.jpg',lane_image)
                    cv2.imwrite('lanes/latest.jpg', lane_image)
                    
                    # count_marker_ = self.sum_markers(lane_image)
                    count_marker_ = x1-x0
                    self.red_marker = filter_red(lane_image).shape[0]>1
                    # print(self.red_marker)

                    if count_marker_ != -1: # == -1  when it is likely not a lane divider, or simply cannot count
                        all_markers_count.append(count_marker_)
        
        all_markers_count = np.array(all_markers_count)
        if len(all_markers_count) == 0: return
        
        current_speed = scipy.stats.mode(all_markers_count).mode * self.marker_size / (times[-1]-times[i]) * self.fps
        if self.current_speed == FrameDataConst.UNKNOWN:
            self.current_speed = current_speed
        else:
            self.pct_change = (current_speed-self.current_speed)/self.current_speed
            self.current_speed = current_speed
        if round(self.current_speed,2) > 800:
            print('>>>> time = {}, {}'.format(times[-1],times[i]))
            print('>>>> all_markers_count = {}'.format(all_markers_count))
            print(f'>>>> current_speed = {self.current_speed}')
            print(f'>>>> pct_change = {self.pct_change}')
            print(f'>>>> marker_size_pixel = {self.marker_size_pixel}')
            print(anchor_list)
            raise

    def sum_markers(self, image:np.ndarray) -> float:
        """ First, divide the input image into different segments by colors. Then pass each one-color segment into count_markers to count the number of markers on that particular segment. Finally, sum all the markers of all segments with different colors together. 

        Args:
            image (np.ndarray): image of a segment of the lane divider, might contain different colors in one.

        Returns:
            float: the number of markers counted on the segment
        """
        # make sure the image is always vertical
        height, width, _ = image.shape
        # print(image.shape)
        if height < width: # horizontal image
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            height, width = width, height

        return height/self.marker_size_pixel

        # get markers based on common colors
        red_marker = filter_red(image)
        yellow_marker = filter_yellow(image)
        blue_marker = filter_blue(image)
        
        # filtered_markers len should be 1 or 2
        filtered_markers = []
        if red_marker.shape[0] > 1: 
            filtered_markers.append(red_marker)
            self.red_marker = True
        if yellow_marker.shape[0] > 1: filtered_markers.append(yellow_marker)
        if blue_marker.shape[0] > 1: filtered_markers.append(blue_marker)

        # print(f'len(filtered_markers) = {len(filtered_markers)}')
        # print(f'width={width}, height={height}, marker_size={self.marker_size_pixel}')
        
        if len(filtered_markers) == 0: return -1
        if len(filtered_markers) > 2: return -1 # likely this is not a lane divider, model's error
        if len(filtered_markers) == 2: # there are 2 colors in one divider segment, use self.marker_size_pixel (calculated from 1-color segment)
            if self.marker_size_pixel != -1: return height/self.marker_size_pixel
            else: return -1
        
        # only one color
        # counted_markers = self.count_markers(filtered_markers[0])
        return height/self.marker_size_pixel

        # return counted_markers
                
    
    def count_markers(self, image:np.ndarray) -> float:
        """ Count number of markers per image by dividing the height of the image by the frequency of the image.
        The image is a segment of lane divider with only one color.

        Args:
            image (np.ndarray): image of a segment of a lane divider in one color

        Returns:
            float: the number of markers counted on the segment
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = np.array(gray_image, dtype=float)
        
        # Make sure the image is always vertical
        height, width = gray_image.shape
        if height < width: # horizontal image
            gray_image = cv2.rotate(gray_image, cv2.ROTATE_90_CLOCKWISE)
            height, width = width, height
        
        # Window the image.
        window_x = np.hanning(gray_image.shape[1])
        window_y = np.hanning(gray_image.shape[0])
        gray_image *= np.outer(window_y, window_x)
        # Transform to frequency domain.
        spectrum = np.fft.rfft2(gray_image)
        # Partially whiten the spectrum. This tends to make the autocorrelation sharper,
        # but it also amplifies noise. The -0.6 exponent is the strength of the
        # whitening normalization, where -1.0 would be full normalization and 0.0 would
        # be the usual unnormalized autocorrelation.
        spectrum *= (1e-12 + np.abs(spectrum))**-0.6
        # Exclude some very low frequencies, since these are irrelevant to the texture.
        fx = np.arange(spectrum.shape[1])
        fy = np.fft.fftshift(np.arange(spectrum.shape[0]) - spectrum.shape[0] // 2)
        fx, fy = np.meshgrid(fx, fy)
        spectrum[np.sqrt(fx**2 + fy**2) < 10] = 0
        # Compute the autocorrelation and inverse transform.
        acorr = np.real(np.fft.irfft2(np.abs(spectrum)**2))

        clip_corr = np.clip(acorr, 0, np.percentile(acorr, 99.5))
        peaks, _ = signal.find_peaks(clip_corr[:acorr.shape[0]//2,0])
        
        if len(peaks) == 0: return -1 # cannot find a peak, likely because the image is too small
        
        repeated_size = peaks[0] # the first peak means the pattern will repeat every peaks[0] pixels
        if self.marker_size_pixel == -1: self.marker_size_pixel = repeated_size
        
        return height/repeated_size 
   
        
