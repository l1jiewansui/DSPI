import numpy as np
from scipy.spatial.distance import euclidean
from ..core.Reference import mainRef_Point

class CalculateIValue:
    def __init__(self, cluster, mainRefPoint):
        self.cluster = cluster.clu_point  # Assuming cluster is a Clu_Point instance
        self.mainRef_Point = mainRefPoint
        self.num_refSet = len(mainRefPoint.ref_points)
        self.dim = len(cluster.clu_point[0].coordinate)
        self.num_data = len(cluster.clu_point)
        self.n = 10 ** (self.num_refSet + 1)  # Assuming NUM_CIRCLE - 1 = 9 for simplicity

        self.process()
        self.radix_sorti()
        self.radix_sortp()
        self.radix_sortq()
        self.calc_linear_fun()

    def binary_search(self, disList, target):
        low, high = 0, len(disList) - 1
        while low <= high:
            mid = (low + high) // 2
            if disList[mid] < target:
                low = mid + 1
            elif disList[mid] > target:
                high = mid - 1
            else:
                return mid
        return -1

    def radix_sorti(self):
        # Sort points based on i_value, assuming i_value is directly accessible
        sorted_points = sorted(self.cluster, key=lambda x: x.i_value)
        self.mainRef_Point.set_i_value_pts(sorted_points)

    def radix_sortp(self):
        # Sort points based on i_value, assuming i_value is directly accessible
        sorted_points = sorted(self.cluster, key=lambda x: x.p_value)
        self.mainRef_Point.ref_points[0].set_p_value_pts(sorted_points)
    
    def radix_sortq(self):
        # Sort points based on i_value, assuming i_value is directly accessible
        sorted_points = sorted(self.cluster, key=lambda x: x.q_value)
        self.mainRef_Point.ref_points[1].set_q_value_pts(sorted_points)

    def calc_linear_fun(self):
        # Calculate linear function using numpy, assuming points have a numeric i_value
        # 在构造 A 之前，确保 x 正确格式化
        x = np.array([p.i_value for p in self.cluster], dtype=np.float64)
        # print(x)
        y = np.arange(self.num_data)
        # print(y)
        A = np.vstack([x, np.ones(len(x))]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        self.a = a
        self.b = b

    def calculate_euclidean_dis(self, point_a, point_b):
        # Adjust to use Point instances
        return euclidean(point_a.coordinate, point_b.coordinate)

    def process(self):
        # Adjust to use Point instances and attributes
        for point in self.cluster:
            dis_pt_mainRefPt = self.calculate_euclidean_dis(point, self.mainRef_Point.point)
            rank_cir = self.binary_search(self.mainRef_Point.dis, dis_pt_mainRefPt)
            i_value = rank_cir
            point.i_value = i_value 
            
            if self.mainRef_Point.ref_points:
                ref_point = self.mainRef_Point.ref_points[0]
                dis_pt_refPt = self.calculate_euclidean_dis(point, ref_point.point)
                rank_refCir = self.binary_search(ref_point.dis, dis_pt_refPt)
                # 这里采用进位的方式，通过一个高位的i_value存储了所有枢轴的i_value
                p_value = rank_refCir
                point.p_value = p_value

                ref_point = self.mainRef_Point.ref_points[1]
                dis_pt_refPt = self.calculate_euclidean_dis(point, ref_point.point)
                rank_refCir = self.binary_search(ref_point.dis, dis_pt_refPt)
                # 这里采用进位的方式，通过一个高位的i_value存储了所有枢轴的i_value
                q_value = rank_refCir
                point.q_value = q_value
            

    def get_cluster(self):
        return self.cluster
    
    def get_main_ref_point(self):
        return self.mainRef_point
        
