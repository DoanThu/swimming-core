from dataclasses import dataclass, asdict
from json import dumps
import torch
from typing import List
from dataclasses import field


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

class FrameDataConst:
    UNKNOWN = -1
    LEFT, RIGHT, UP, DOWN = 0, 1, 2, 3
    READY, JUMP, RACE, STOP, TURN, DOLPHIN_KICK = 0, 1, 2, 3, 4, 5
    BACKSTROKE, BUTTERFLY, FREESTYLE, BREASTSTROKE = 0, 1, 2, 3
    VERTICAL, HORIZONTAL = 0, 1
    
    MAP_DIRECTION = {-1: 'UNKNOWN', 0: 'LEFT', 1: 'RIGHT', 2: 'UP', 3: 'DOWN'}
    MAP_STATUS = {-1: 'UNKNOWN', 0: 'READY', 1: 'JUMP', 2: 'RACE', 3: 'STOP', 4: 'TURN', 5: 'DOLPHIN_KICK'}
    MAP_ORIENTATION = {-1: 'UNKNOWN', 0: 'VERTICAL', 1: 'HORIZONTAL'}
    MAP_STROKE = {-1: 'UNKNOWN', 0: 'BACKSTROKE', 1: 'BUTTERFLY', 2: 'FREESTYLE', 3: 'BREASTSTROKE'}

@dataclass
class FrameData(SuperDataClass):
    # frame info
    frame_idx: int = 0
    frame_orientation: int = FrameDataConst.UNKNOWN
    
    # swimmer info
    direction: int = FrameDataConst.UNKNOWN
    status: int = FrameDataConst.READY
    reach_marker: bool = False
    skeleton: list = field(default_factory=list)
    bbox: torch.Tensor = torch.Tensor(0, 4) # xmin, ymin, xmax, ymax
    bbox_area: float = FrameDataConst.UNKNOWN
    stroke: int = FrameDataConst.UNKNOWN
    stroke_count: int = 0
    face_up: bool = False
    
    # speed info 
    period_index: int = FrameDataConst.UNKNOWN
    speed: float = FrameDataConst.UNKNOWN
    speed_pct_change: float = FrameDataConst.UNKNOWN
    red_marker: bool = False
     
    # process info
    pose_time: float = FrameDataConst.UNKNOWN
    segment_time: float = FrameDataConst.UNKNOWN
    total_time: float = FrameDataConst.UNKNOWN
    
    def __str__(self) -> List[str]:
        """ Return a list of key:value pairs in string format for debugging

        Returns:
            List[str]: List of key:value pairs in string format
        """
        direction_str = f'direction:{FrameDataConst.MAP_DIRECTION[self.direction]}'
        status_str = f'status:{FrameDataConst.MAP_STATUS[self.status]}'
        speed_str = f'speed:{self.speed:.2f}'
        pct_change_str = f'pct_change:{self.speed_pct_change:+.2f}'
        orientation_str = f'orientation:{FrameDataConst.MAP_ORIENTATION[self.frame_orientation]}'
        red_marker_str = f'red_marker:{self.red_marker}'
        return [speed_str, pct_change_str]