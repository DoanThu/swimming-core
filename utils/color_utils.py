import cv2
import numpy as np


def crop_image_from_mask(image: np.ndarray, mask: np.ndarray,
                         threshold:float=0.4) -> np.ndarray:
    """ Select image given its mask. If the mask contains more than one area, the area with the largest bbox area will be selected. This function is use for cropping the lane divider's images only.
    Have not tested with horizontal image.

    Args:
        image (np.ndarray): 2D image, shape = (height, width, 3). Better be a long rectangle.
        mask (np.ndarray): binary image, shape = (height, width). values range from 0 to 255.
        threshold (float, optional): percentage of points of the mask per row or per column, depending on the input image. Defaults to 0.4.

    Returns:
        np.ndarray: 2D array. shape = (new_height, width, 3) or shape = (height, new_width, 3), depending on the input image
    """
    mask = np.array(mask/mask.max(),dtype=np.uint8)

    height, width = mask.shape
    axis = 1 if height > width else 0
        
    valids = np.where(np.mean(mask,axis=axis)>threshold)[0]
    if len(valids) == 0: return np.array([])
    arr = [valids[0]]
    max_len = 0
    final_arr = []
    for i in range(1,len(valids)):
        if valids[i] - 1 == arr[-1]:
            arr.append(valids[i])
        else:
            if max_len < len(arr):
                max_len = len(arr)
                final_arr = arr
            arr = [valids[i]]
    if max_len < len(arr):
        max_len = len(arr)
        final_arr = arr
    return image[final_arr[0]:final_arr[-1],:,:] if height > width else image[:,final_arr[0]:final_arr[-1],:]
            
    

def filter_red(image: np.ndarray) -> np.ndarray:
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # lower mask (0-10)
    lower_red = np.array([0,0,0])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170,0,0])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
    
    mask = mask0+mask1
    return mask 
    if np.all(mask==0): return np.array([])
    return crop_image_from_mask(image, mask)


def filter_blue(image: np.ndarray) -> np.ndarray:
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # mask
    lower_blue = np.array([90, 0, 0]) 
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
    return mask
    
    if np.all(mask==0): return np.array([])
    return crop_image_from_mask(image, mask)
 


def filter_yellow(image: np.ndarray) -> np.ndarray:
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # mask
    lower_yellow = np.array([12, 0, 0])
    upper_yellow = np.array([32, 255, 255])
    mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    return mask
    
    if np.all(mask==0): return np.array([])
    return crop_image_from_mask(image, mask)

