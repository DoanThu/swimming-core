import cv2 
import torch 
import numpy as np
from typing import List
import colorsys

# The skeleton is in 17-keypoint YOLO format
SKELETON = [[[0,1], [1,3], [5,7], [7,9], [11,13], [13,15]], # left color
           [[0,2], [2,4], [6,8], [8,10], [12,14], [14,16]], # right color
           [[5,6], [5,11], [6,12], [11,12]], # another color
           ]

def draw_keypoints(image:np.ndarray, keypoints:torch.Tensor, thickness:int=1) -> np.ndarray:
    """ Draw body keypoints on image

    Args:
        image (np.ndarray): 2-d image
        keypoints (torch.Tensor): Keypoints with shape = (N, 17, 2)

    Returns:
        np.ndarray: Return image with annotations
    """
    colors = [(0,0,255),(255,0,0),(0,255,0)] # red is left, blue is right, green is the torso
    for _keypoints in keypoints:
        for i,v in enumerate(SKELETON):
            for j in v:
                _from, _to = j[0], j[1]
                if _keypoints[_from][0] == 0 and _keypoints[_from][1] == 0:
                    continue
                if _keypoints[_to][0] == 0 and _keypoints[_to][1] == 0:
                    continue
                cv2.line(image, (int(_keypoints[_from][0]), int(_keypoints[_from][1])),
                        (int(_keypoints[_to][0]), int(_keypoints[_to][1])), colors[i], thickness=thickness)
    return image


def write_single_text(image:np.ndarray, text:str, 
               font:str=cv2.FONT_HERSHEY_SIMPLEX, 
               color:tuple=(0, 255, 0), org:tuple=(50,50), 
               font_scale:float=1.0, thickness:int=2, text_color_bg:tuple=(0, 0, 0)) -> np.ndarray:
    """ Write a line of text into given image and return that image with text

    Args:
        image (np.ndarray): image
        text (str): one line of text
        font (str, optional): font type. Defaults to cv2.FONT_HERSHEY_SIMPLEX.
        color (tuple, optional): color . Defaults to (0, 255, 0), which is green.
        org (tuple, optional): coordinate of the text. Defaults to (50,50).
        font_scale (int, optional): font scale. Defaults to 1.
        thickness (int, optional): thickness. Defaults to 2.
        text_color_bg (tuple, optional): background color. Defaults to (0, 0, 0), which is black.

    Returns:
        np.ndarray: image with text
    """
    x, y = org
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    
    cv2.rectangle(image, org, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(image, text, (x, int(y + text_h + font_scale - 1)), font, font_scale, color, thickness,  cv2.LINE_AA)
    return image


def write_texts(image:np.ndarray, texts:List[str], spacing:int, 
               font:str=cv2.FONT_HERSHEY_SIMPLEX, 
               color:tuple=(0, 255, 0), org:tuple=(50,50), 
               font_scale:float=1.0, thickness:int=2, text_color_bg:tuple=(0, 0, 0)) -> np.ndarray:
    """ Write multiple lines of text into given image and return the image. All lines are left aligned.

    Args:
        image (np.ndarray): image
        texts (List[str]): list of text, each in one line
        spacing (int): space between 2 adjacent lines
        font (str, optional): font. Defaults to cv2.FONT_HERSHEY_SIMPLEX.
        color (tuple, optional): color. Defaults to (0, 255, 0), which is green.
        org (tuple, optional): coordinate of the first line of text. Defaults to (50,50).
        font_scale (int, optional): font scale. Defaults to 1.
        thickness (int, optional): thickness. Defaults to 2.
        text_color_bg (tuple, optional): color of the background. Defaults to (0, 0, 0), which is black.

    Returns:
        np.ndarray: image with multiple lines of text
    """
    orgx, orgy = org
    for text in texts:
        write_single_text(image, text, font, color, 
                          (orgx, orgy), font_scale, 
                          thickness, text_color_bg)
        orgy += spacing
    return image


def draw_segmentation(image:np.ndarray, list_segments: list,
                      color:tuple=(255,0,0), thickness:int=1) -> np.ndarray:
    """ Draw polygons listed in list_segments onto image

    Args:
        image (np.ndarray): 2D image
        list_segments (list): list of segments. Each segment contains a list of points
        color (tuple, optional): color of the polygons. Defaults to (255,0,0).
        thickness (int, optional): thickness of the polygons. Defaults to 1.

    Returns:
        np.ndarray: image with segments 
    """
    for segment in list_segments:
        pts = np.array(segment, dtype=int)
        pts = pts.reshape((-1, 1, 2))

        image = cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)
    return image


def draw_detection(image:np.ndarray, list_boxes: list,
                      color:tuple=(0,255,0), thickness:int=1) -> np.ndarray:

    for x,y,w,h in list_boxes:
        x,y,w,h = int(x), int(y), int(w), int(h)
        start_point = (x,y)
        end_point = (x+w, y+h)

        image = cv2.rectangle(image, start_point, end_point, color, thickness)
    return image


def draw_dot(image:np.ndarray, dots:np.ndarray,
             radius=2, color=(0,0,255), thickness=-1, 
             frame_idx='', font:str=cv2.FONT_HERSHEY_SIMPLEX, 
             font_scale:int=0.5, thickness_txt:int=1
             ) -> np.ndarray:
    """ Draw multiple dots on image

    Args:
        image (np.ndarray): 2D image
        dots (np.array): list of N dots, shape = (N, 2)
        radius (int, optional): radius of the dot. Defaults to 2.
        color (tuple, optional): corlor of the dot. Defaults to red (0,0,255).
        thickness (int, optional): thickness of the dot. Defaults to -1.

    Returns:
        np.ndarray: image with dots
    """
    for i,dot in enumerate(dots):
        x, y = int(dot[0]),int(dot[1])
        image = cv2.circle(image, (x, y), radius=radius, color=color, thickness=thickness)

        if frame_idx == '': continue
        # draw time on the dot
        cv2.putText(image, frame_idx, (x, y-10), font, fontScale=font_scale, color=color, thickness=thickness_txt)

    return image

		
def generate_even_light_colors(n=30):
    """
    Generate n evenly spaced light RGB colors (pastel-like)
    suitable for dark backgrounds.
    Uses HSV space: evenly spaced hues, high value and low saturation.
    """
    colors = []
    for i in range(n):
        hue = i / n  # evenly spaced hues
        saturation = 0.4  # lower saturation for pastel effect
        value = 0.95      # high brightness for visibility on dark backgrounds
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    return np.array(colors)