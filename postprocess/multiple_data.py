from dataclasses import dataclass, asdict
from json import dumps
import torch
from typing import List
from dataclasses import field
from postprocess.const import FrameDataConst

@dataclass
class SuperDataClass:
    @property
    def __dict__(self):
        """
        get a python dictionary
        """
        return asdict(self)

    @property
    def json(self):
        """
        get the json formated string
        """
        return dumps(self.__dict__)

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
    bbox_list: torch.Tensor = torch.empty(0, 4) # bbox of the skeletons (xmin, ymin, xmax, ymax)
    bbox_area_list: list = field(default_factory=list)
    stroke_list: list = field(default_factory=list)
    stroke_count_list: list = field(default_factory=list)
    face_up_list: list = field(default_factory=list)
    
    # speed info 
    period_index_list: list = field(default_factory=list)
    speed_m_list: list = field(default_factory=list) # speed in meters
    speed_px_list: list = field(default_factory=list) # speed in pixels
    speed_pct_change_list: list = field(default_factory=list) 
    # red_marker_list: list = field(default_factory=list)
     
    # process info
    pose_time_list: list = field(default_factory=list)
    segment_time_list: list = field(default_factory=list)
    total_time_list: list = field(default_factory=list)
    
    def __str__(self) -> List[str]:
        """ Return a list of key:value pairs in string format for debugging

        Returns:
            List[str]: List of key:value pairs in string format
        """
        # direction_str = f'direction:{FrameDataConst.MAP_DIRECTION[self.direction]}'
        # status_str = f'status:{FrameDataConst.MAP_STATUS[self.status]}'
        speed_m_str = f'speed_m_list:{self.speed_m_list:.2f}'
        speed_px_str = f'speed_px_list:{self.speed_px_list:.2f}'
        pct_change_str = f'speed_pct_change_list:{self.speed_pct_change_list:+.2f}'
        # orientation_str = f'orientation:{FrameDataConst.MAP_ORIENTATION[self.frame_orientation]}'
        # red_marker_str = f'red_marker:{self.red_marker}'
        return [speed_m_str, speed_px_str, pct_change_str]