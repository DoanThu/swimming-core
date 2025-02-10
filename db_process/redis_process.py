import struct
import numpy as np
import json
import cv2
import time

def image_to_redis(r,a,n):
    """Store given Numpy array 'a' in Redis under key 'n'"""
    h, w = a.shape[:2]
    #    shape = struct.pack('>II',h,w)
    #    encoded = shape + a.tobytes()

    start = time.time()
    (flag, encoded_img) = cv2.imencode(".jpg", a)
    if not flag:
        return

    encoded_img = encoded_img.tobytes()

    # Store encoded data in Redis
    r.set(n, encoded_img)
    end = time.time()
    # print(f'Elapsed time for image_to_redis: {end-start}')
    


def json_to_redis(conn, data, key):
      conn.set(key, data)

def frame_data_to_redis(conn, frame_data, key):
    d = {'speed': frame_data.speed, 'pct_change': frame_data.speed_pct_change}
    json_to_redis(conn, json.dumps(d), key)

def dict_to_redis(conn, d, key, limit=5):
    keylist = list(d.keys())[:limit]
    newd = {k:d[k] for k in keylist}
    json_to_redis(conn, json.dumps(newd), key)