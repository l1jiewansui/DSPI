import numpy as np
import math

class CalCirclePos:
    def __init__(self, refPt, radius_refPt, queryPt, radius_queryPt):
        self.dim = len(refPt.point.coordinate)
        self.dis_upper = None
        self.dis_lower = None
        self.label = None
        self.calculate_position(refPt, radius_refPt, queryPt, radius_queryPt)

    def calculate_position(self, refPt, radius_refPt, queryPt, radius_queryPt):
        # 使用 NumPy 进行距离计算
        distance = self.calculate_distance_of_points(refPt.point, queryPt)
        
        # 优化后的判断逻辑，减少分支条件的复杂性
        total_radius = radius_refPt + radius_queryPt
        radius_diff = abs(radius_refPt - radius_queryPt)

        if distance >= total_radius:  # 外切+外离
            self.label = 1
            self.dis_lower = float('inf')
            self.dis_upper = float('inf')
        elif distance >= radius_diff:  # 相交
            self.label = 2
            self.dis_lower = max(0.0, distance - radius_queryPt)
            self.dis_upper = radius_refPt
        else:  # 内含
            self.label = 3
            self.dis_lower = max(0.0, distance - radius_queryPt)
            self.dis_upper = distance + radius_queryPt

    def calculate_distance_of_points(self, point_a, point_b):
        # 使用 NumPy 进行向量化距离计算
        coord_a = np.array(point_a.coordinate)
        coord_b = np.array(point_b.coordinate)
        return np.linalg.norm(coord_a - coord_b)
