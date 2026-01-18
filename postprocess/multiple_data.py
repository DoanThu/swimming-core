from dataclasses import dataclass, asdict
from json import dumps
import torch
from typing import List
from dataclasses import field
from postprocess.const import FrameDataConst

@dataclass
class SuperDataClass:
    @property
    def json(self):
        """
        get the json formated string
        """
        return dumps(asdict(self))

@dataclass
class FrameMultipleData(SuperDataClass):
    # frame info
    frame_idx: int = 0
    frame_orientation: int = FrameDataConst.UNKNOWN
    
    # swimmer info
    swimmer_id_list : list = field(default_factory=list) # list of swimmer ids
    direction_list: list = field(default_factory=list)
    status_list: list = field(default_factory=list)
    skeleton_list: list = field(default_factory=list)
    bbox_list: list = field(default_factory=list)
    bbox_area_list: list = field(default_factory=list)
    stroke_list: list = field(default_factory=list)
    stroke_rate_list: list = field(default_factory=list)
    face_up_list: list = field(default_factory=list)
    
    # speed info 
    period_index_list: list = field(default_factory=list)
    speed_m_list: list = field(default_factory=list) # speed in meters
    speed_px_list: list = field(default_factory=list) # speed in pixels
    speed_pct_change_list: list = field(default_factory=list) 
    distance_per_stroke_list: list = field(default_factory=list)
     
    # process info
    pose_time: float = FrameDataConst.UNKNOWN
    segment_time: float = FrameDataConst.UNKNOWN
    speed_calculation_time: float = FrameDataConst.UNKNOWN
    anchor_update_time: float = FrameDataConst.UNKNOWN
    visualization_time: float = FrameDataConst.UNKNOWN
    tracking_time: float = FrameDataConst.UNKNOWN
    count_stroke_time: float = FrameDataConst.UNKNOWN
    analysis_time: float = FrameDataConst.UNKNOWN
    total_time: float = FrameDataConst.UNKNOWN
    
    def __str__(self) -> List[str]:
        """ Return a list of key:value pairs in string format for debugging

        Returns:
            List[str]: List of key:value pairs in string format
        """
        # direction_str = f'direction:{FrameDataConst.MAP_DIRECTION[self.direction]}'
        # status_str = f'status:{FrameDataConst.MAP_STATUS[self.status]}'
        speed_m_str = f'speed_m_list:{self.speed_m_list}'
        speed_px_str = f'speed_px_list:{self.speed_px_list}'
        pct_change_str = f'speed_pct_change_list:{self.speed_pct_change_list}'
        # orientation_str = f'orientation:{FrameDataConst.MAP_ORIENTATION[self.frame_orientation]}'
        # red_marker_str = f'red_marker:{self.red_marker}'
        return [speed_m_str, speed_px_str, pct_change_str]