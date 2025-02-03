import numpy as np 

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
        return np.sqrt((ax-bx)**2+(ay-by)**2)
    
    def angle(self,ax,ay,bx,by,cx,cy):
        import math
        from math import degrees
        ang = degrees(math.atan2(cy-by, cx-bx) - math.atan2(ay-by, ax-bx))
        return ang + 360 if ang < 0 else ang

    def extract_distance_features(self, skeleton):
        # skeleton is a 3-level nested loop
        final_res = {}
        for i in range(self.n_landmarks):
            for j in range(i+1,self.n_landmarks):
                # x, y
                ax, ay = skeleton[0][i][0], skeleton[0][i][1]
                bx, by = skeleton[0][j][0], skeleton[0][j][1]
                colname = self.MAP_KEYPOINTS[i] + '_' + self.MAP_KEYPOINTS[j]
                final_res[colname] = self.distance(ax,ay,bx,by)
        if self.cur_dist_features:
            self.prev_dist_features = self.cur_dist_features
        self.cur_dist_features = final_res
        
    
    def extract_angle_features(self, skeleton):
        # skeleton is a 3-level nested loop
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
        if self.cur_angle_features:
            self.prev_angle_features = self.cur_angle_features
        self.cur_angle_features = final_res

    def extract_pct_change(self):
        if not self.prev_angle_features: return
        # distance
        for i in range(self.n_landmarks):
            for j in range(i+1,self.n_landmarks):
                colname = self.MAP_KEYPOINTS[i] + '_' + self.MAP_KEYPOINTS[j]
                self.pct_dist_changes[colname] = (self.cur_dist_features[colname] - self.prev_dist_features[colname])/self.prev_dist_features[colname]
        self.pct_dist_changes = dict(sorted(self.pct_dist_changes.items(), key=lambda item: abs(item[1]), 
                                            reverse=True))

        # angle
        for i in range(self.n_landmarks):
            for j in range(i+1,self.n_landmarks):
                for k in range(j+1,self.n_landmarks):
                    colname = self.MAP_KEYPOINTS[i] + '_' + self.MAP_KEYPOINTS[j] + '_' + self.MAP_KEYPOINTS[k]
                    self.pct_angle_changes[colname] = self.cur_angle_features[colname] - self.prev_angle_features[colname]
        self.pct_angle_changes = dict(sorted(self.pct_angle_changes.items(), key=lambda item: abs(item[1]), 
                                             reverse=True))

