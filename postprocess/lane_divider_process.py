import numpy as np
from postprocess.single_data import FrameData, FrameDataConst
from model_caller.seg_model_caller import SegCaller
from scipy import signal
import cv2


class LaneDivider:
    def __init__(self):
        self.orientation = FrameDataConst.UNKNOWN
        self.unit_size = 1

    def set_lane_divider_info(self, frame: np.ndarray,
                              seg_caller: SegCaller, **seg_kwargs):
        full_lane_dividers = seg_caller.get_lane_dividers(frame, **seg_kwargs)
        if len(full_lane_dividers) == 0: 
            return 
        dividers_size = [len(i) for i in full_lane_dividers]
        full_lane_divider = full_lane_dividers[np.argmax(dividers_size)]
        xmax, xmin = np.max(full_lane_divider[:,0]), np.min(full_lane_divider[:,0])
        ymax, ymin = np.max(full_lane_divider[:,1]), np.min(full_lane_divider[:,1])

        # print(f'direction = {xmax-xmin, ymax-ymin}')
        # print(full_lane_divider)
        # set orientation
        if xmax-xmin > ymax-ymin:
            self.orientation = FrameDataConst.HORIZONTAL
        else:
            self.orientation = FrameDataConst.VERTICAL

        # set unit size (in pixel)
        # xmax, xmin = int(xmax), int(xmin)
        # ymax, ymin = int(ymax), int(ymin)
        # lane_image = frame[ymin:ymax,xmin:xmax,:]
        # self.set_unit_size(lane_image)
        
    
    def set_unit_size(self, image:np.ndarray) -> float:
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
        
        if len(peaks) == 0: self.unit_size = -1 # cannot find a peak, likely because the image is too small
        
        # the first peak means the pattern will repeat every peaks[0] pixels
        self.unit_size = peaks[0]
        