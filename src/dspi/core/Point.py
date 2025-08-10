import numpy as np
import math

class Point:
    def __init__(self, coordinate=None, id=None, i_value=None, p_value=None, q_value=None, is_deleted=False):
        self.coordinate = coordinate if coordinate is not None else []
        self.id = id
        self.i_value = i_value
        self.p_value = p_value
        self.q_value = q_value
        self.is_deleted = is_deleted 

    def set_i_value(self, i_value):
        self.i_value = i_value
    
    def set_p_value(self, p_value):
        self.p_value = p_value

    def set_q_value(self, q_value):
        self.q_value = q_value

    def delete(self):
        self.is_deleted = False

    def __repr__(self):
        return f"Point ID: {self.id}, Coordinates: {self.coordinate}, Deleted: {self.is_deleted}"

class InsertPt:
    def __init__(self, coordinate=None, id=None, i_value=None):
        self.coordinate = coordinate if coordinate is not None else []
        self.id = id
        self.i_value = i_value

    def set_i_value(self, i_value):
        self.i_value = i_value

class Clu_Point:
    def __init__(self, clu_point=None):
        self.clu_point = clu_point if clu_point is not None else []

class All_Point:
    def __init__(self, all_point=None):
        self.all_point = all_point if all_point is not None else []