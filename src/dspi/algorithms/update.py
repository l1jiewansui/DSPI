import pickle
import math
from tqdm import tqdm
from Point import Point
import numpy as np
import time
from PointQuery import PointQuery
import uuid

def load_index(file_path):
    start_time = time.perf_counter()
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    elapsed_time = (time.perf_counter() - start_time) * 1000
    print(f"Index loaded in {elapsed_time:.2f} ms.")
    return data

def calculate_euclidean_distance(point_a, point_b):
    coord_a = np.array(point_a.coordinate)
    coord_b = np.array(point_b.coordinate)
    return np.linalg.norm(coord_a - coord_b)

def find_closest_centroid(all_refSet, new_point):
    distances = np.array([calculate_euclidean_distance(ref.point, new_point) for ref in all_refSet])
    min_index = np.argmin(distances)
    closest_centroid = all_refSet[min_index].point
    closest_cluster = all_refSet[min_index]
    return closest_centroid, closest_cluster

def insert_point_into_cluster(cluster, new_point):
    cluster.iValuePts.append(new_point)
    cluster.is_sorted = False

def exists_in_clusters(all_refSet, new_point):
    for ref in all_refSet:
        point_query = PointQuery(new_point, ref)
        print(point_query.query_res)
        if point_query.query_res:
            return True
    return False

def delete_in_clusters(all_refSet, new_point):
    found = False
    for ref in all_refSet:
        point_query = PointQuery(new_point, ref)
        if point_query.query_res:
            found = True
            break
    return found

def insert_point(all_refSet, new_point):
    new_point_obj = Point(new_point)
    if not exists_in_clusters(all_refSet, new_point_obj):
        closest_centroid, closest_cluster = find_closest_centroid(all_refSet, new_point_obj)
        dist_qt_Ref = calculate_euclidean_distance(new_point_obj, closest_centroid)
        pred_id = closest_cluster.coeffs(dist_qt_Ref)
        new_point_obj.id = int(pred_id)
        print(new_point_obj.id)
        insert_point_into_cluster(closest_cluster, new_point_obj)
        update_distances(closest_cluster, new_point_obj, dist_qt_Ref)
    else:
        print("Point already exists in a cluster.")

def update_distances(cluster, new_point, dist_qt_Ref):
    for ref_point in cluster.ref_points:
        new_distance = calculate_euclidean_distance(ref_point.point, new_point)
        ref_point.dis.append(new_distance)
        ref_point.dis.sort()
    main_distance = dist_qt_Ref
    cluster.dis.append(main_distance)
    cluster.dis.sort()

def delete_point(all_refSet, point_to_delete):
    new_point_obj = Point(point_to_delete)
    if delete_in_clusters(all_refSet, new_point_obj):
        for refSet in all_refSet:
            for point in refSet.iValuePts:
                if np.array_equal(point.coordinate, point_to_delete):
                    point.is_deleted = True
                    return
            for ref_point in refSet.ref_points:
                for point in ref_point.pValuePts:
                    if np.array_equal(point.coordinate, point_to_delete):
                        point.is_deleted = True
                        return
                for point in ref_point.qValuePts:
                    if np.array_equal(point.coordinate, point_to_delete):
                        point.is_deleted = True
                        return
        elapsed_time = (time.perf_counter() - time.perf_counter()) * 1000
        print(f"Point {point_to_delete} deleted in {elapsed_time:.2f} ms.")
    else:
        print("Point not found.")

def save_updated_index(all_refSet, file_path):
    start_time = time.perf_counter()
    with open(file_path, 'wb') as file:
        pickle.dump({'all_refSet': all_refSet}, file)
    elapsed_time = (time.perf_counter() - start_time) * 1000
    print(f"Index updated and saved to {file_path} in {elapsed_time:.2f} ms.")