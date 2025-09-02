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

def get_ground(img):
    # Convert to HSV color space
    height, width, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define HSV range for yellow-beige ground (you can tweak these)
    lower_yellow = np.array([15, 40, 120])
    upper_yellow = np.array([35, 180, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_beige = np.array([0, 0, 180])
    upper_beige = np.array([40, 60, 255])
    mask_beige = cv2.inRange(hsv, lower_beige, upper_beige) 

    # ---- Combine and Clean Masks ----
    combined_mask = cv2.bitwise_or(mask_yellow, mask_beige)

    # Morphological operations to clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_DILATE, kernel)

    # Find contours from the mask
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours (e.g., people, noise)
    min_area = 3000
    filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Combine contours into one bounding box
    if filtered:
        # Combine all points into one array
        all_points = np.vstack(filtered)

        # Get the unified bounding box
        x, y, w, h = cv2.boundingRect(all_points)

        if w > h:
            return (-1,)
        if w*img.shape[0] > width*height/2:
            return (-1, )
        return (x,0,w,img.shape[0])
    else:
        return (-1,)

def bbox_overlap_or_near(bbox1, bbox2, threshold=30):
    """
    Check if bbox1 and bbox2 either intersect or are within threshold pixels of each other.
    bbox = (x, y, w, h)
    """

    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    left1, right1, top1, bottom1 = x1, x1 + w1, y1, y1 + h1
    left2, right2, top2, bottom2 = x2, x2 + w2, y2, y2 + h2

    # Check for intersection
    intersect = not (right1 < left2 or right2 < left1 or bottom1 < top2 or bottom2 < top1)
    if intersect:
        return True

    dist1 = min(abs(left1 - right2), abs(left2 - right1))
    return dist1 <= threshold