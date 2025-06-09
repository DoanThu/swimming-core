import numpy as np 
from collections import defaultdict
from config.general import REAL_SIZE_HEIGHT, REAL_SIZE_WIDTH

class ExtractParams:
    MAP_KEYPOINTS = {0:'nose', 1:'leye', 2:'reye', 3:'lear', 4:'rear', 5:'lshoulder', 6:'rshoulder', 7:'lelbow', 8:'relbow',
                     9:'lwrist', 10:'rwrist', 11:'lhip', 12:'rhip', 13:'lknee', 14:'rknee', 15:'lankle', 16:'rankle'}
    def __init__(self, n_landmarks=17):
        self.n_landmarks = n_landmarks

        self.cur_dist_features = {}
        self.prev_dist_features = {}
        self.cur_angle_features = {}
        self.prev_angle_features = {}

        self.pct_dist_changes = {}
        self.pct_angle_changes = {}

    def distance(self,ax,ay,bx,by):
        return np.sqrt(((ax-bx)*REAL_SIZE_WIDTH)**2+((ay-by)*REAL_SIZE_HEIGHT)**2)
    
    def angle(self,ax,ay,bx,by,cx,cy):
        import math
        from math import degrees
        ang = degrees(math.atan2(cy-by, cx-bx) - math.atan2(ay-by, ax-bx))
        return abs(ang)

    def extract_distance_features(self, skeleton):
        # skeleton is a 3-level nested loop
        if not skeleton: return
        final_res = {}
        for i in range(self.n_landmarks):
            for j in range(i+1,self.n_landmarks):
                # x, y
                ax, ay = skeleton[0][i][0], skeleton[0][i][1]
                bx, by = skeleton[0][j][0], skeleton[0][j][1]
                colname = self.MAP_KEYPOINTS[i] + '_' + self.MAP_KEYPOINTS[j]
                final_res[colname] = self.distance(ax,ay,bx,by)
        return final_res

    
    def extract_angle_features(self, skeleton):
        # skeleton is a 3-level nested loop
        if not skeleton: return
        final_res = {}
        for i in range(self.n_landmarks):
            for j in range(i+1,self.n_landmarks):
                for k in range(j+1,self.n_landmarks):
                    # x, y
                    ax, ay = skeleton[0][i][0], skeleton[0][i][1]
                    bx, by = skeleton[0][j][0], skeleton[0][j][1]
                    cx, cy = skeleton[0][k][0], skeleton[0][k][1]
                    colname = self.MAP_KEYPOINTS[i] + '_' + self.MAP_KEYPOINTS[j] + '_' + self.MAP_KEYPOINTS[k]
                    final_res[colname] = self.angle(ax,ay,bx,by,cx,cy)
        return final_res

    # def extract_pct_change(self):
    #     if not self.prev_angle_features: return
    #     # distance
    #     for i in range(self.n_landmarks):
    #         for j in range(i+1,self.n_landmarks):
    #             colname = self.MAP_KEYPOINTS[i] + '_' + self.MAP_KEYPOINTS[j]
    #             self.pct_dist_changes[colname] = (self.cur_dist_features[colname] - self.prev_dist_features[colname])/self.prev_dist_features[colname]
    #     self.pct_dist_changes = dict(sorted(self.pct_dist_changes.items(), key=lambda item: abs(item[1]), 
    #                                         reverse=True))

    #     # angle
    #     for i in range(self.n_landmarks):
    #         for j in range(i+1,self.n_landmarks):
    #             for k in range(j+1,self.n_landmarks):
    #                 colname = self.MAP_KEYPOINTS[i] + '_' + self.MAP_KEYPOINTS[j] + '_' + self.MAP_KEYPOINTS[k]
    #                 self.pct_angle_changes[colname] = self.cur_angle_features[colname] - self.prev_angle_features[colname]
    #     self.pct_angle_changes = dict(sorted(self.pct_angle_changes.items(), key=lambda item: abs(item[1]), 
    #                                          reverse=True))


    def extract_pct_change_list(self, prev_features_list, cur_features_list):
        # prev_features_list = [{distance, angle}, {distance, angle}, ...]
        # cur_features_list = [{distance, angle}, {distance, angle}, ...]
        prev_distance = defaultdict(list)
        prev_angle = defaultdict(list)
        for item in prev_features_list:
            for k,v in item[0].items():
                prev_distance[k].append(v)
            for k,v in item[1].items():
                prev_angle[k].append(v)
        for k in prev_distance.keys():
            self.prev_dist_features[k] = np.mean(prev_distance[k])
        for k in prev_angle.keys():
            self.prev_angle_features[k] = np.mean(prev_angle[k])

        cur_distance = defaultdict(list)
        cur_angle = defaultdict(list)
        for item in cur_features_list:
            for k,v in item[0].items():
                cur_distance[k].append(v)
            for k,v in item[1].items():
                cur_angle[k].append(v)
        for k in cur_distance.keys():
            self.cur_dist_features[k] = np.mean(cur_distance[k])
        for k in cur_angle.keys():
            self.cur_angle_features[k] = np.mean(cur_angle[k])
        
        for k in prev_distance.keys():
            self.pct_dist_changes[k] = (self.cur_dist_features[k] - self.prev_dist_features[k])/self.prev_dist_features[k]
        for k in prev_angle.keys():
            self.pct_angle_changes[k] = self.cur_angle_features[k] - self.prev_angle_features[k]
        
        self.pct_dist_changes = dict(sorted(self.pct_dist_changes.items(), key=lambda item: abs(item[1]), 
                                            reverse=True))
        self.pct_angle_changes = dict(sorted(self.pct_angle_changes.items(), key=lambda item: abs(item[1]), 
                                             reverse=True))