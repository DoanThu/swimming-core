import numpy as np
from postprocess.multiple_data import FrameMultipleData
from utils.visualize_utils import draw_keypoints, write_texts, draw_segmentation, draw_dot, draw_detection, generate_even_light_colors

RANDOM_COLORS = np.random.permutation(generate_even_light_colors(n=30))

def visualize_results(frame: np.ndarray, 
                      frame_data: FrameMultipleData,
                      lane_divider_bboxes: list,
                      bbox_ground: list,
                      anchor_list: dict,
                      debug: bool = False,
                      show_frame_ixd: bool = False,
                      from_socket: bool = False
                      ) -> np.ndarray:
    
    annotated_frame = frame.copy()
    annotated_frame = draw_keypoints(annotated_frame, frame_data.skeleton_list, thickness=1)
    
    if from_socket:
        for i in range(len(frame_data.swimmer_id_list)):
            swimmer_id = frame_data.swimmer_id_list[i]
            skel = frame_data.skeleton_list[i]
            annotated_txt = f'ID:{swimmer_id}'
            annotated_frame = write_texts(annotated_frame, [annotated_txt], 10, 
                                        org=(int(skel[0][0]), int(skel[0][1])),
                                        font_scale=0.5, color=RANDOM_COLORS[swimmer_id%len(RANDOM_COLORS)].tolist()
                                        )
        return annotated_frame

    if debug:
        annotated_frame = draw_detection(annotated_frame, lane_divider_bboxes) # draw bbox for lane dividers

    if debug:
        if len(bbox_ground) != 0 and bbox_ground[0] != -1:
            annotated_frame = draw_detection(annotated_frame, [bbox_ground])
        texts = frame_data.__str__()
        annotated_frame = write_texts(annotated_frame, texts, 30, org=(30,30))

    # draw anchor points    
    if debug:
        for k,v in anchor_list.items():
            if show_frame_ixd:
                frame_idx_ = str(k)
            else:
                frame_idx_ = ''
            point_list = np.array([ap.coord for ap in v])
            swimmer_ids = np.array([ap.swimmer_id for ap in v])
            for point, swimmer_id in zip(point_list, swimmer_ids):
                annotated_frame = draw_dot(annotated_frame, [point], 
                                            color=RANDOM_COLORS[swimmer_id%len(RANDOM_COLORS)].tolist(),
                                            radius=5,
                                            frame_idx=frame_idx_)
        
    # visualize swimmer id, speed and stroke count
    for i in range(len(frame_data.swimmer_id_list)):
        swimmer_id = frame_data.swimmer_id_list[i]

        skel = frame_data.skeleton_list[i]
        spd = frame_data.speed_m_list[i] 
        stroke_count = frame_data.stroke_count_list[i]
        distance_per_stroke = frame_data.distance_per_stroke_list[i]

        annotated_txt = f'ID:{swimmer_id}, {spd:.2f}m/s, {stroke_count} spm, {distance_per_stroke} dps'
        annotated_frame = write_texts(annotated_frame, [annotated_txt], 10, 
                                    org=(int(skel[0][0]), int(skel[0][1])),
                                    font_scale=0.5, color=RANDOM_COLORS[swimmer_id%len(RANDOM_COLORS)].tolist()
                                    )
    return annotated_frame

def visualize_swimmer_id(frame: np.ndarray, frame_data: FrameMultipleData) -> np.ndarray:
    annotated_frame = frame.copy()
    for i in range(len(frame_data.swimmer_id_list)):
        swimmer_id = frame_data.swimmer_id_list[i]
        skel = frame_data.skeleton_list[i]
        annotated_txt = f'ID:{swimmer_id}'
        annotated_frame = write_texts(annotated_frame, [annotated_txt], 10, 
                                    org=(int(skel[0][0]), int(skel[0][1])),
                                    font_scale=0.5, color=RANDOM_COLORS[swimmer_id%len(RANDOM_COLORS)].tolist()
                                    )
    return annotated_frame