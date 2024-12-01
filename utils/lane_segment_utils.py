import numpy as np 
import cv2

def is_vertical(points: np.ndarray) -> bool:
    """Return True if the 2-D points form a vertical bounding box

    Args:
        points (np.ndarray): the polygon that covers the lane divider returned by the segmentation model

    Returns:
        bool: bool
    """
    xmin, xmax = np.min(points[:,0]), np.max(points[:,0])
    ymin, ymax = np.min(points[:,1]), np.max(points[:,1])
    if xmax - xmin > 5 * (ymax - ymin):
        return False
    return True


def get_contour_red_obj(image: np.ndarray) -> np.ndarray:
    """Return the mask to filter red color

    Args:
        image (np.ndarray): original RGB image

    Returns:
        np.ndarray: contours of any red objects
    """
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join masks
    mask = mask0+mask1

    # apply mask and get contour
    output_img = image.copy()
    output_img[np.where(mask==0)] = 0
    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    return contours 
    


def get_red_marker(points: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Return the contour of the red marker

    Args:
        points (np.ndarray): the polygon that covers the lane divider returned by the segmentation model
        image (np.ndarray): the original image

    Returns:
        np.ndarray: the contour of the red marker
    """
    points = np.array(points, dtype=np.int32)
    filled_mask = np.zeros((image.shape[0], image.shape[1]))
    cv2.fillConvexPoly(filled_mask, points, 1)
    filled_mask = filled_mask > 0 # To convert to Boolean
    lane_image = np.zeros_like(image)
    lane_image[filled_mask] = image[filled_mask]
    return get_contour_red_obj(lane_image)

