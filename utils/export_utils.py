# This file is temporary until we make a proper package

from pathlib import Path
from collections.abc import Iterable

def _coerce_point(pt):
    """Accept (x,y,v) tuples/lists or dicts with x/y/(v|visibility|c)."""
    if isinstance(pt, dict):
        x = pt.get('x', 0)
        y = pt.get('y', 0)
        v = pt.get('v', pt.get('visibility', pt.get('c', 0)))
    else:
        x = pt[0] if len(pt) > 0 else 0
        y = pt[1] if len(pt) > 1 else 0
        v = pt[2] if len(pt) > 2 else 0
    try:    x = float(x)
    except: x = 0.0
    try:    y = float(y)
    except: y = 0.0
    try:    v = int(round(float(v)))
    except: v = 0
    return x, y, 1

def _normalize_17_points(skeleton):
    """Pad/truncate to 17 keypoints."""
    pts = []
    for pt in skeleton:
        pts.append(_coerce_point(pt))
        if len(pts) == 17:
            break
    if len(pts) < 17:
        pts += [(0.0, 0.0, 0)] * (17 - len(pts))
    return pts

def skeleton_to_row(skeleton, tail_value, decimals=2, sep=" "):
    """
    Flatten one skeleton (17 points) to a single CSV-like row:
    x1,y1,v1,...,x17,y17,v17,tail
    """
    pts = _normalize_17_points(skeleton)
    nums = []
    for (x, y, v) in pts:
        nums.extend([f"{x:.{decimals}f}", f"{y:.{decimals}f}", str(v)])
    nums.append(str(tail_value))
    return sep.join(nums)

# This format is for some paper writing
def save_skeleton_rows(skeletons, out_path, tail=0, decimals=2, sep=" "):
    """
    Write one or many skeleton rows to a .txt file.
    - `skeletons` can be a single skeleton or a list of skeletons.
    - `tail` can be a single value (applied to all) or a list aligned with skeletons.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize to list of skeletons
    if isinstance(skeletons, dict) or (isinstance(skeletons, Iterable) and skeletons and not isinstance(skeletons[0], Iterable)):
        skeletons = [skeletons]

    # Normalize tail(s)
    if isinstance(tail, Iterable) and not isinstance(tail, (str, bytes)):
        tails = list(tail)
        if len(tails) != len(skeletons):
            raise ValueError("Length of `tail` list must match number of skeletons.")
    else:
        tails = [tail] * len(skeletons)

    with open(out_path, "w", encoding="utf-8") as f:
        for skel, tv in zip(skeletons, tails):
            line = skeleton_to_row(skel, tv, decimals=decimals, sep=sep)
            f.write(line + "\n")
