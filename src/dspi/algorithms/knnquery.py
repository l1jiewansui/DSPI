import math
import numpy as np
from typing import List, Tuple
from ..core.Point import Point
from ..core.Reference import mainRef_Point, Ref_Point
from ..utils import Constants
from .CalCirclePos import CalCirclePos

class KNN:
    def __init__(self, queryPoint: Point, radius: float, mainRef_point: mainRef_Point, k: int):
        self.queryPoint = queryPoint
        self.k = k
        self.mainRef_point = mainRef_point
        self.rangeQueryRes: List[int] = []
        self.distance: List[float] = []
        self.sorted_result: List[Tuple[float, int]] = []

        self.mainRefPtCircle = CalCirclePos(mainRef_point, mainRef_point.r, queryPoint, radius)
        self.ref_query = [CalCirclePos(rp, rp.r, queryPoint, radius) for rp in mainRef_point.ref_points]

        self._ensure_arrays()
        selected, r_best = self._knn_pipeline()
        self._final_calc(selected, r_best)

    def _ensure_arrays(self) -> None:
        mr = self.mainRef_point
        if not hasattr(mr, "coords_array"):
            mr.coords_array = np.array([p.coordinate for p in mr.iValuePts]) if getattr(mr, "iValuePts", None) else np.empty((0, 0))
            mr.ids_array = np.array([p.id for p in mr.iValuePts]) if getattr(mr, "iValuePts", None) else np.array([])
        for rp in mr.ref_points:
            if not hasattr(rp, "coords_array"):
                pts = rp.pValuePts if getattr(rp, "pValuePts", None) else (rp.qValuePts if getattr(rp, "qValuePts", None) else [])
                if pts:
                    rp.coords_array = np.array([p.coordinate for p in pts])
                    rp.ids_array = np.array([p.id for p in pts])
                else:
                    dim = mr.coords_array.shape[1] if mr.coords_array.size else 0
                    rp.coords_array = np.empty((0, dim))
                    rp.ids_array = np.array([])
            if hasattr(rp, "dis"):
                rp.dis = np.sort(np.asarray(rp.dis))
        if hasattr(mr, "dis"):
            mr.dis = np.sort(np.asarray(mr.dis))

    def _clu_sort(self) -> List[Tuple[Ref_Point, float, float]]:
        pivots = [(self.mainRef_point, self.mainRefPtCircle)] + list(zip(self.mainRef_point.ref_points, self.ref_query))
        scores = []
        eps = 1e-9
        for ref_pt, circle in pivots:
            dq = np.linalg.norm(np.asarray(self.queryPoint.coordinate) - np.asarray(ref_pt.point.coordinate))
            width = max(0.0, float(circle.dis_upper - circle.dis_lower))
            width_norm = width / (float(ref_pt.dis[-1]) + eps) if getattr(ref_pt, "dis", None) is not None and len(ref_pt.dis) else 1.0
            rho = 1.0 / (dq + eps) * 1.0 / (1.0 + width_norm)
            scores.append((ref_pt, rho, dq))
        total = sum(r for _, r, _ in scores) + 1e-12
        scores = [(rp, r / total, dq) for rp, r, dq in scores]
        scores.sort(key=lambda t: t[1], reverse=True)
        top_m = min(self.k, len(scores))
        return scores[:top_m]

    @staticmethod
    def _count_band(dis: np.ndarray, center: float, r: float) -> int:
        if dis.size == 0:
            return 0
        lo = max(0.0, center - r)
        hi = center + r
        left = int(np.searchsorted(dis, lo, side="left"))
        right = int(np.searchsorted(dis, hi, side="right"))
        return max(0, right - left)

    def _adap_rad_opt(self, selected: List[Tuple[Ref_Point, float, float]]) -> float:
        # binary search for r so that weighted expected count >= k
        if not selected:
            return 0.0
        # upper bound: max(dq)+max(dis)
        r_hi = 0.0
        for rp, _, dq in selected:
            max_dis = float(rp.dis[-1]) if getattr(rp, "dis", None) is not None and len(rp.dis) else 0.0
            r_hi = max(r_hi, dq + max_dis)
        r_lo = 0.0
        target = self.k
        for _ in range(40):
            mid = 0.5 * (r_lo + r_hi)
            exp_cnt = 0.0
            for rp, rho, dq in selected:
                dis = np.asarray(rp.dis) if getattr(rp, "dis", None) is not None else np.array([])
                cnt = self._count_band(dis, dq, mid)
                exp_cnt += rho * cnt
            if exp_cnt >= target:
                r_hi = mid
            else:
                r_lo = mid
        return r_hi

    def _gather_candidates(self, selected: List[Tuple[Ref_Point, float, float]], r: float) -> Tuple[np.ndarray, np.ndarray]:
        q = np.asarray(self.queryPoint.coordinate)
        ids_list: List[np.ndarray] = []
        coords_list: List[np.ndarray] = []
        for rp, _, dq in selected:
            if not hasattr(rp, "coords_array") or rp.coords_array.size == 0:
                continue
            dis = np.asarray(rp.dis) if getattr(rp, "dis", None) is not None else np.array([])
            if dis.size == 0:
                continue
            lo = max(0.0, dq - r)
            hi = dq + r
            left = int(np.searchsorted(dis, lo, side="left"))
            right = int(np.searchsorted(dis, hi, side="right"))
            sl = slice(left, right)
            coords_list.append(rp.coords_array[sl])
            ids_list.append(rp.ids_array[sl])
        if not coords_list:
            return np.empty((0, q.shape[0])), np.array([], dtype=int)
        coords = np.vstack(coords_list)
        ids = np.concatenate(ids_list)
        # deduplicate by id, keep first occurrence
        uniq_ids, idx = np.unique(ids, return_index=True)
        return coords[idx], uniq_ids

    def _final_calc(self, selected: List[Tuple[Ref_Point, float, float]], r_best: float) -> None:
        coords, ids = self._gather_candidates(selected, r_best)
        if coords.size == 0:
            self.sorted_result = []
            self.rangeQueryRes = []
            self.distance = []
            return
        q = np.asarray(self.queryPoint.coordinate)
        dists = np.linalg.norm(coords - q, axis=1)
        order = np.argsort(dists)[: self.k]
        top_ids = ids[order]
        top_d = dists[order]
        self.sorted_result = [(float(top_d[i]), int(top_ids[i])) for i in range(len(order))]
        self.rangeQueryRes = [int(i) for i in top_ids]
        self.distance = [float(x) for x in top_d]

    def _knn_pipeline(self) -> Tuple[List[Tuple[Ref_Point, float, float]], float]:
        selected = self._clu_sort()
        r_best = self._adap_rad_opt(selected)
        return selected, r_best

    def predict_bounds(self, ref_point, circle):
        try:
            # 确保 dis_lower 和 dis_upper 的值有效
            if not (circle.dis_lower >= 0 and circle.dis_upper >= 0):
                print("Invalid circle.dis_lower or circle.dis_upper")
                return 0, 0  # 返回一个默认的范围值
            
            lower = ref_point.coeffs[0] + sum(ref_point.coeffs[i] * circle.dis_lower ** i for i in range(1, Constants.COEFFS + 1))
            upper = ref_point.coeffs[0] + sum(ref_point.coeffs[i] * circle.dis_upper ** i for i in range(1, Constants.COEFFS + 1))
            
            if lower > upper:  # 确保 lower <= upper
                lower, upper = upper, lower
            
            lower_bound = max(0, min(len(ref_point.dis) - 1, int(lower)))
            upper_bound = max(0, min(len(ref_point.dis) - 1, int(upper)))
        except ValueError as e:
            print(f"Error calculating bounds: {e}")
            lower_bound, upper_bound = 0, 0  # 使用默认范围值
        
        return lower_bound, upper_bound
    
    # def predict_bounds(self, ref_point, circle):
    #     # 假设 circle.dis_lower 和 circle.dis_upper 是需要预测边界的输入
    #     # 并且 ref_point.model 是已经训练好的神经网络模型
    #     redundancy = 0
    #     try:
    #         # 确保 dis_lower 和 dis_upper 的值有效
    #         if not (circle.dis_lower >= 0 and circle.dis_upper >= 0):
    #             print("Invalid circle.dis_lower or circle.dis_upper")
    #             return 0, 0  # 返回一个默认的范围值
    #         with torch.no_grad():  # 确保在预测时不计算梯度
    #             model = ref_point.model  # 获取模型
    #             inputs_lower = torch.tensor([[circle.dis_lower]]).float()  # 转换为适合模型的输入格式
    #             inputs_upper = torch.tensor([[circle.dis_upper]]).float()

    #             predicted_lower = model(inputs_lower).item()  # 进行预测并转换为Python数值
    #             predicted_upper = model(inputs_upper).item()

    #             # print(predicted_lower, predicted_upper)

    #         # 预测的下界和上界可能需要根据实际情况进行调整
    #         lower_bound = max(0, min(len(ref_point.dis) - 1, int(predicted_lower) - redundancy))
    #         upper_bound = max(0, min(len(ref_point.dis) - 1, int(predicted_upper) + redundancy))
    #     except ValueError as e:
    #         # print(f"Error calculating bounds: {e}")
    #         lower_bound, upper_bound = 0, 0  # 使用默认范围值
        
    #     return lower_bound, upper_bound

    def doubling_search(self, disList, is_forward, target):
        """实现倍增搜索逻辑"""
        n = len(disList)
        if is_forward:
            step = 1
            index = 0
            while index < n and disList[index] < target:
                index += step
                step *= 2
            return min(index, n - 1)
        else:
            step = 1
            index = n - 1
            while index >= 0 and disList[index] > target:
                index -= step
                step *= 2
            return max(index, 0)

    def binary_search_for_doubling(self, disList, target, low, high):
        """实现对倍增搜索结果的二分搜索"""
        while low <= high:
            mid = (low + high) // 2
            if disList[mid] < target:
                low = mid + 1
            elif disList[mid] > target:
                high = mid - 1
            else:
                return mid
        return low if disList[low] >= target else high

    # def do_knn_query(self, rank_list, mainRef_point, K):
    #     # 使用集合来存储结果，确保唯一性
    #     results_set = set()

    #     # 遍历每个参考点与查询点之间的关系
    #     for ref_circle in self.ref_query:
    #         if ref_circle.label == 1:
    #             continue  # 如果 label==1，跳过当前参考点

    #     # 如果没有跳过，则处理当前参考点
    #     if rank_list:
    #         ids_set_1 = set()
    #         rank_range = rank_list[0]
    #         lower_bound, upper_bound = rank_range
    #         for idx in range(lower_bound, upper_bound + 1):
    #             point = mainRef_point.iValuePts[idx]
    #             ids_set_1.add(point.id)

    #         ids_set_2 = set()
    #         rank_range = rank_list[1]
    #         lower_bound, upper_bound = rank_range
    #         for idx in range(lower_bound, upper_bound + 1):
    #             point = mainRef_point.ref_points[0].pValuePts[idx]
    #             ids_set_2.add(point.id)

    #         ids_set_3 = set()
    #         rank_range = rank_list[2]
    #         lower_bound, upper_bound = rank_range
    #         for idx in range(lower_bound, upper_bound + 1):
    #             point = mainRef_point.ref_points[1].qValuePts[idx]
    #             ids_set_3.add(point.id)

    #         points_dict_1 = {point.id: point for point in mainRef_point.iValuePts}

    #         # 计算三个集合的交集
    #         # intersection_ids = ids_set_1.union(ids_set_2, ids_set_3)
    #         intersection_ids = ids_set_1.intersection(ids_set_2, ids_set_3)

    #         # 对交集中的ID对应的点计算距离并更新结果集
    #         results_set = set()
    #         for id_ in intersection_ids:
    #             point = points_dict_1[id_]
    #             distance = self.calculate_euclidean_distance(self.queryPoint, point)
    #             if distance <= self.radius:
    #                 results_set.add((distance, point.id))

    #     # 根据距离排序结果集合，转换为列表以便排序
    #     sorted_results = sorted(list(results_set), key=lambda x: x[0])

    #     # 选择前 K 个结果的点的ID，如果结果数小于 K，则返回所有结果
    #     if len(sorted_results) <= K:
    #         self.sorted_result = sorted_results
    #         self.rangeQueryRes = [result[1] for result in sorted_results]
    #         self.distance = [result[0] for result in sorted_results]
    #     else:
    #         self.sorted_result = sorted_results[:K]
    #         self.rangeQueryRes = [result[1] for result in sorted_results[:K]]
    #         self.distance = [result[0] for result in sorted_results[:K]]

    def do_knn_query(self, rank_list, mainRef_point, K):

        # method_start_time = time.perf_counter()  # 方法开始时记录时间
        # 使用集合来自动处理重复项
        results_set = set()
        results_list = []

        # 首先检查主参考点与查询点之间的关系，如果 label==1，则跳过
        if self.mainRefPtCircle.label == 1:
            # print("Skipping range query for mainRefPtCircle due to label==1")
            self.rangeQueryRes = sorted(list(results_set))
            return

        # 遍历每个参考点与查询点之间的关系
        for ref_circle in self.ref_query:
            if ref_circle.label == 1:
                # print(f"Skipping range query for ref_point due to label==1: {ref_circle}")
                self.rangeQueryRes = sorted(list(results_set))
                return

        # process_rank_list_start_time = time.perf_counter()  # 处理排名列表前记录时间

        if rank_list:
            rank_range = rank_list[0]
            lower_bound, upper_bound = rank_range
            
            # 直接从mainRef_point.iValuePts提取ID和坐标，避免后续重复提取
            coords_slice = mainRef_point.dict_circle[lower_bound:upper_bound+1]
            ids_slice = mainRef_point.insert_list[0][lower_bound:upper_bound+1]

            # end_time_ids_set = time.perf_counter()
            # print(f"Time for creating ID sets and extracting coordinates: {(end_time_ids_set - start_time_ids_set)*1000} ms")

            # start_time_distance = time.perf_counter()
            query_point_coords = np.array(self.queryPoint.coordinate)

            #######################打印检查了多少个点#####################
            # total_considered_points = len(coords_slice)
            # print(f"Total points considered in this query: {total_considered_points}")

            # 计算所有点与查询点之间的距离
            distances = np.linalg.norm(coords_slice - query_point_coords, axis=1)
            within_radius_indices = np.where(distances <= self.radius)[0]

            # 更新结果集，这里假设结果集需要的是点的ID
            # results_ids = ids_slice[within_radius_indices]
            # results_set.update(results_ids)

            # 更新结果集，这里假设结果集需要的是点坐标
            # 如果找到的点少于K个，则直接返回这些点
            if len(within_radius_indices) <= K:
                for index in within_radius_indices:
                    results_list.append(coords_slice[index])
            else:
                # 否则，找到距离最近的K个点
                nearest_indices = np.argsort(distances)[:K]
                for index in nearest_indices:
                    results_list.append(coords_slice[index])

            # end_time_distance = time.perf_counter()
            # print(f"Time for calculating distances and updating result set: {(end_time_distance - start_time_distance)*1000} ms")

        # process_rank_list_end_time = time.perf_counter()  # 处理排名列表后记录时间
        # print(f"Processing Rank List Time: {(process_rank_list_end_time - process_rank_list_start_time)*1000} ms")

        # # 将结果集合转换为列表并排序
        self.sorted_result = results_list

    def search_positions(self, disList, target_distance):
        """使用倍增搜索和二分搜索找到目标距离的位置"""
        if target_distance - self.radius < disList[0]:
            lower_bound = 0
        else:
            lower_bound = self.doubling_search(disList, True, target_distance - self.radius)
            lower_bound = self.binary_search_for_doubling(disList, target_distance - self.radius, 0, lower_bound)
        
        if target_distance + self.radius > disList[-1]:
            upper_bound = len(disList) - 1
        else:
            upper_bound = self.doubling_search(disList, False, target_distance + self.radius)
            upper_bound = self.binary_search_for_doubling(disList, target_distance + self.radius, upper_bound, len(disList) - 1)

        return lower_bound, upper_bound
    
    def calculate_euclidean_distance(self, point_a, point_b):
        # 计算欧几里得距离
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point_a.coordinate, point_b.coordinate)))
    
    def binary_search(self, iValuePts, target, find_low):
        """
        实现二分搜索找到满足条件的点的范围。
        :param iValuePts: 参考点的值数组
        :param target: 搜索目标值
        :param find_low: 如果为True，查找满足条件的最低索引；如果为False，查找满足条件的最高索引
        :return: 索引位置
        """
        low, high = 0, len(iValuePts) - 1
        result = -1
        while low <= high:
            mid = (low + high) // 2
            if find_low:
                if iValuePts[mid].i_value >= target:
                    result = mid
                    high = mid - 1
                else:
                    low = mid + 1
            else:
                if iValuePts[mid].i_value <= target:
                    result = mid
                    low = mid + 1
                else:
                    high = mid - 1
        return result