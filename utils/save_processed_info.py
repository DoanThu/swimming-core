from dataclasses import asdict, is_dataclass, fields
from pathlib import Path
import pandas as pd
import numpy as np
import cv2, json
# from config.general import SAVE_ANALYSIS_PATH

def _is_simple(x):
    return isinstance(x, (int, float, str, bool, type(None), np.integer, np.floating))

def _json_default(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def dataclass_to_row(dc_obj, flatten=True, prefix=""):
    """
    Convert a dataclass (possibly nested) to a flat dict (if flatten=True),
    else return {"meta_json": "..."} with JSON-serialized content.
    """
    assert is_dataclass(dc_obj), "Expected a dataclass instance"
    d = asdict(dc_obj)

    if not flatten:
        return {"meta_json": json.dumps(d, default=_json_default)}  # store whole object as JSON

    # Flatten: only keep simple scalars as columns; JSON-pack complex fields
    row = {}
    for k, v in d.items():
        col = f"{prefix}{k}"
        if _is_simple(v):
            row[col] = v
        else:
            # print(f'col={col}')
            # pack lists, dicts, nested structures as JSON
            if isinstance(v, np.ndarray):
                v = v.tolist()
            row[col] = json.dumps(v, default=_json_default)
    return row

def save_video_and_index(
    frames,              # iterable of (frame_idx, frame_bgr_np)
    metas,               # list aligned with frames OR dict keyed by frame_idx
    out_dir,
    fps=30.0,
    video_name="video.mp4",
    write_thumbs=True,
    thumb_width=320,
    flatten_meta=True,   # set False to store a single 'meta_json' column
):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    thumbs_dir = out/"thumbs"
    if write_thumbs: thumbs_dir.mkdir(exist_ok=True)

    frames = list(frames)
    h, w = frames[0][1].shape[:2]
    vw = cv2.VideoWriter(str(out/video_name), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    rows = []
    for i, (idx, img_bgr) in enumerate(frames):
        vw.write(img_bgr)

        thumb_rel = None
        if write_thumbs:
            scale = thumb_width / w
            thumb = cv2.resize(img_bgr, (thumb_width, int(h*scale)))
            p = thumbs_dir/f"{idx:06d}.jpg"
            cv2.imwrite(str(p), thumb)
            thumb_rel = p.relative_to(out).as_posix()

        meta = metas[idx] if isinstance(metas, dict) else metas[i]
        # meta must be a dataclass instance
        meta_cols = dataclass_to_row(meta, flatten=flatten_meta)

        rows.append({
            "frame_adjusted_idx": int(idx),
            "t_ms": (idx / fps) * 1000.0,
            "video_path": video_name,
            "thumb_path": thumb_rel,
            **meta_cols
        })

    vw.release()
    df = pd.DataFrame(rows).sort_values("frame_adjusted_idx").reset_index(drop=True)
    if 'swimmer_id_list' not in df.columns: # track single swimmer
        df['swimmer_id_list'] = ['[0]']*len(df)
    if 'stroke_count_list' not in df.columns:
        df['stroke_count_list'] = df.apply(lambda row: '[' + str(row['stroke_count']) + ']', axis=1)
    if 'speed_m_list' not in df.columns:
        df['speed_m_list'] = df.apply(lambda row: '[' + str(row['speed_m']) + ']', axis=1)
    if 'distance_per_stroke_list' not in df.columns:
        df['distance_per_stroke_list'] = df.apply(lambda row: '[' + str(row['distance_per_stroke']) + ']', axis=1)
    if 'bbox_list' not in df.columns:
        df['bbox_list'] = df.apply(lambda row: '[' + str(row['bbox']) + ']', axis=1)
    

    df.to_parquet(out/"metadata.parquet", index=False)


def load_index(out_dir):
    return pd.read_parquet(Path(out_dir)/"metadata.parquet")

def row_to_dataclass(row, meta_cls, flatten_meta=True):
    """
    Recreate a dataclass from a DataFrame row.
    - If flattened: pick fields by name from row; JSON-decode if needed.
    - If JSON mode: read row['meta_json'].
    """
    assert is_dataclass(meta_cls), "meta_cls must be a dataclass type"

    if not flatten_meta:
        data = json.loads(row["meta_json"])
        return meta_cls(**data)

    # Flatten mode: pull every declared field from the row by name.
    # If a field isn't a simple type, try JSON-decoding it.
    kwargs = {}
    for f in fields(meta_cls):
        name = f.name
        val = row.get(name, None)  # assumes flat names == field names
        if isinstance(val, str) and val and (val.startswith("{") or val.startswith("[")):
            try:
                val = json.loads(val)
            except Exception:
                pass
        kwargs[name] = val
    return meta_cls(**kwargs)
