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

def image_from_redis(r,n):
        """Retrieve Numpy array from Redis key 'n'"""
        encoded = r.get(n)
        h, w = struct.unpack('>II',encoded[:8])
        a = np.frombuffer(encoded, dtype=np.uint8, offset=8).reshape(h,w,3)
        return a

def dict_to_redis(conn,data,key):
      conn.hmset(key, data)

def json_to_redis(conn, data, key):
      conn.set(key, data)

def frame_data_to_redis(conn, frame_data, key):
    d = {'speed': frame_data.speed, 'pct_change': frame_data.pct_change}
    json_to_redis(conn, json.dumps(d), key)
