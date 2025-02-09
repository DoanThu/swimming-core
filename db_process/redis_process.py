import struct
import numpy as np
import json

def image_to_redis(r,a,n):
   """Store given Numpy array 'a' in Redis under key 'n'"""
   h, w = a.shape[:2]
   shape = struct.pack('>II',h,w)
   encoded = shape + a.tobytes()

   # Store encoded data in Redis
   r.set(n,encoded)
   return


def json_to_redis(conn, data, key):
      conn.set(key, data)

def frame_data_to_redis(conn, frame_data, key):
    d = {'speed': frame_data.speed, 'pct_change': frame_data.speed_pct_change}
    json_to_redis(conn, json.dumps(d), key)

def dict_to_redis(conn, d, key, limit=5):
    keylist = list(d.keys())[:limit]
    newd = {k:d[k] for k in keylist}
    json_to_redis(conn, json.dumps(newd), key)