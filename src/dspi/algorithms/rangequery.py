import numpy as np
import time
from functools import lru_cache
from dataclasses import dataclass, field
from typing import List
import math
import heapq
from ..core.Point import Point
from ..core.Reference import Ref_Point
from ..core.Reference import mainRef_Point

class RangeQuery:

    def __init__(self, query_point, radius: float, main_ref: mainRef_Point):
        self.query_point = query_point
        self.radius = radius
        self.main_ref = main_ref
        self.range_query_res: set[int] = set()

        self.main_circle = CalCirclePos(main_ref, main_ref.r, query_point, radius)
        self.sub_circles = [
            CalCirclePos(rp, rp.r, query_point, radius) for rp in main_ref.ref_points
        ]

        self._prepare_rank_list()
        self._do_range_query()

    def _prepare_rank_list(self):
        ref_pairs = [(self.main_ref, self.main_circle)] + list(zip(self.main_ref.ref_points, self.sub_circles))
        heap = []
        for ref_pt, circle in ref_pairs:
            lb, ub = self._predict_bounds(ref_pt, circle)
            if lb <= ub:
                heapq.heappush(heap, (ub - lb, lb, ub, id(ref_pt), ref_pt))
        self.rank_list = [(lb, ub, ref) for _, lb, ub, _, ref in heapq.nsmallest(len(heap), heap)]

    def _predict_bounds(self, ref_pt: Ref_Point, circle):
        if not (circle.dis_lower >= 0 and circle.dis_upper >= 0):
            return 0, 0
        try:
            lower_pred, upper_pred = ref_pt.model([circle.dis_lower, circle.dis_upper])
        except Exception:
            return 0, 0
        if np.isnan(lower_pred) or np.isnan(upper_pred):
            return 0, 0
        if lower_pred > upper_pred:
            lower_pred, upper_pred = upper_pred, lower_pred
        lb = self._correct_position(ref_pt.dis, lower_pred, True)
        ub = self._correct_position(ref_pt.dis, upper_pred, False)
        return lb, ub

    @staticmethod
    def _correct_position(dis_list, pred_pos, is_lower):
        n = len(dis_list)
        if math.isinf(pred_pos) or math.isnan(pred_pos):
            return 0 if is_lower else n - 1
        pred_pos = max(0, min(n - 1, int(pred_pos)))
        return pred_pos

    def _do_range_query(self):
        if self.main_circle.label == 1 or any(c.label == 1 for c in self.sub_circles):
            return
        lb, ub, ref_pt = min(self.rank_list, key=lambda t: t[1] - t[0])
        if lb > ub:
            return
        q_coords = np.asarray(self.query_point.coordinate)
        pts_coords = ref_pt.coords_array[lb : ub + 1]
        pts_ids = ref_pt.ids_array[lb : ub + 1]
        if pts_coords.size == 0:
            return
        sq_dist = np.sum((pts_coords - q_coords) ** 2, axis=1)
        within = sq_dist <= self.radius ** 2
        self.range_query_res.update(pts_ids[np.nonzero(within)[0]])

from .CalCirclePos import CalCirclePos


def create_test_points_from_file(data_file):
    data = np.genfromtxt(data_file, delimiter=',')
    points = [Point(row.tolist()) for row in data]
    return points


@lru_cache(maxsize=None)
def cached_cal_circle_pos(refSet_id, r_refSet, query_pt_id, r):
    refSet = id_to_refSet[refSet_id]
    query_pt = id_to_query_pt[query_pt_id]
    return CalCirclePos(refSet, r_refSet, query_pt, r)


def perform_range_queries(test_points, all_refSet, r, all_data):
    total_time = 0
    cal_circle_time = 0
    query_time = 0
    total_found_points = 0

    for refSet in all_refSet:
        refSet.coords_array = np.array([pt.coordinate for pt in refSet.iValuePts], dtype=np.float64) if getattr(refSet, 'iValuePts', None) else np.empty((0, 0), dtype=np.float64)
        refSet.ids_array = np.array([pt.id for pt in refSet.iValuePts], dtype=np.int64) if getattr(refSet, 'iValuePts', None) else np.array([], dtype=np.int64)
        if hasattr(refSet, 'dis'):
            refSet.dis = np.sort(np.asarray(refSet.dis))
        for rp in getattr(refSet, 'ref_points', []):
            pts_list = rp.pValuePts if getattr(rp, 'pValuePts', None) else (rp.qValuePts if getattr(rp, 'qValuePts', None) else [])
            if pts_list:
                rp.coords_array = np.array([pt.coordinate for pt in pts_list], dtype=np.float64)
                rp.ids_array = np.array([pt.id for pt in pts_list], dtype=np.int64)
            else:
                dim = refSet.coords_array.shape[1] if refSet.coords_array.size else 0
                rp.coords_array = np.empty((0, dim), dtype=np.float64)
                rp.ids_array = np.array([], dtype=np.int64)
            if hasattr(rp, 'dis'):
                rp.dis = np.sort(np.asarray(rp.dis))

    precomputed_refSets = []
    for refSet in all_refSet:
        refSet_data = {
            'refSet': refSet,
            'r': refSet.r,
        }
        precomputed_refSets.append(refSet_data)

    global id_to_refSet, id_to_query_pt
    id_to_refSet = {id(refSet_data['refSet']): refSet_data['refSet'] for refSet_data in precomputed_refSets}
    id_to_query_pt = {id(pt): pt for pt in test_points}

    for i, query_pt in enumerate(test_points):
        start_time = time.perf_counter()
        total_query_results = []

        query_pt_id = id(query_pt)

        for refSet_data in precomputed_refSets:
            refSet_id = id(refSet_data['refSet'])
            r_refSet = refSet_data['r']

            start_circle_time = time.perf_counter()
            circle = cached_cal_circle_pos(refSet_id, r_refSet, query_pt_id, r)
            end_circle_time = time.perf_counter()
            cal_circle_time += (end_circle_time - start_circle_time)

            if circle.label == 1:
                continue

            start_query_time = time.perf_counter()
            range_query_instance = RangeQuery(query_pt, r, refSet_data['refSet'])
            total_query_results.extend(range_query_instance.range_query_res)
            end_query_time = time.perf_counter()
            query_time += (end_query_time - start_query_time)

        end_time = time.perf_counter()
        query_time_for_this_point = (end_time - start_time) * 1000
        num_points_for_this_query = len(total_query_results)

        total_time += (end_time - start_time)
        total_found_points += num_points_for_this_query

    num_queries = len(test_points)
    average_time = total_time / num_queries if num_queries else 0.0
    average_cal_circle_time = cal_circle_time / num_queries if num_queries else 0.0
    average_query_time = query_time / num_queries if num_queries else 0.0
    average_found_points = total_found_points / num_queries if num_queries else 0.0

    print(f"Average number of points found per query: {average_found_points}")
    print(f"Average Query Time: {average_time * 1000:.2f} ms")
    print(f"Average Calculation Circle Time: {average_cal_circle_time * 1000:.2f} ms")
    print(f"Average Range Query Time: {average_query_time * 1000:.2f} ms")

    return {
        'avg_n': float(average_found_points),
        'avg_time_ms': float(average_time * 1000),
        'avg_calc_circle_ms': float(average_cal_circle_time * 1000),
        'avg_range_query_ms': float(average_query_time * 1000),
    }