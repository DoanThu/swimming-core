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
class FrameData(SuperDataClass):
    # frame info
    frame_idx: int = 0
    frame_orientation: int = FrameDataConst.UNKNOWN
    
    # swimmer info
    direction: int = FrameDataConst.UNKNOWN
    status: int = FrameDataConst.READY
    # reach_marker: bool = False
    skeleton: list = field(default_factory=list)
    # bbox: torch.Tensor = torch.Tensor(0, 4) # bbox of the skeleton (xmin, ymin, xmax, ymax)
    bbox: list = field(default_factory=list)
    bbox_area: float = FrameDataConst.UNKNOWN
    stroke: int = FrameDataConst.UNKNOWN
    stroke_count: int = 0
    face_up: bool = False
    
    # speed info 
    period_index: int = FrameDataConst.UNKNOWN
    speed_m: float = FrameDataConst.UNKNOWN # speed in meters
    speed_px: float = FrameDataConst.UNKNOWN # speed in pixels
    speed_pct_change: float = FrameDataConst.UNKNOWN
    red_marker: bool = False
    distance_per_stroke: float = 0
     
    # process info
    pose_time: float = FrameDataConst.UNKNOWN
    segment_time: float = FrameDataConst.UNKNOWN
    speed_calculation_time: float = FrameDataConst.UNKNOWN
    anchor_update_time: float = FrameDataConst.UNKNOWN
    visualization_time: float = FrameDataConst.UNKNOWN
    total_time: float = FrameDataConst.UNKNOWN
    
    def __str__(self) -> List[str]:
        """ Return a list of key:value pairs in string format for debugging

        Returns:
            List[str]: List of key:value pairs in string format
        """
        direction_str = f'direction:{FrameDataConst.MAP_DIRECTION[self.direction]}'
        status_str = f'status:{FrameDataConst.MAP_STATUS[self.status]}'
        speed_m_str = f'speed_m:{self.speed_m:.2f}'
        speed_px_str = f'speed_px:{self.speed_px:.2f}'
        pct_change_str = f'pct_change:{self.speed_pct_change:+.2f}'
        orientation_str = f'orientation:{FrameDataConst.MAP_ORIENTATION[self.frame_orientation]}'
        red_marker_str = f'red_marker:{self.red_marker}'
        return [speed_m_str, speed_px_str, pct_change_str]