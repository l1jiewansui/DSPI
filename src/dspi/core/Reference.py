from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from .Point import Point 

@dataclass
class Ref_Point:
    point: Optional[Point] = None
    r: float = 0.0
    r_low: float = 0.0
    dis: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))  # 使用 NumPy 数组
    dict_circle: List = field(default_factory=list) 
    pValuePts: List = field(default_factory=list)
    qValuePts: List = field(default_factory=list)
    model: Optional[object] = None 
    coeffs: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))  # 使用 NumPy 数组

    def set_dis_arr(self, dis):
        self.dis = dis

    def set_dict(self, dict_circle):
        self.dict_circle = dict_circle

    def set_p_value_pts(self, pValuePts):
        self.pValuePts = pValuePts

    def set_q_value_pts(self, qValuePts):
        self.qValuePts = qValuePts

    def set_model(self, model):
        self.model = model

    def set_coeffs(self, coeffs):
        self.coeffs = coeffs

@dataclass
class mainRef_Point:
    point: Point = field(default_factory=Point)
    r: float = 0.0
    r_low: float = 0.0
    ref_points: List[Ref_Point] = field(default_factory=list)
    dis: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))  # 使用 NumPy 数组
    iValuePts: List = field(default_factory=list)
    coeffs: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))  # 使用 NumPy 数组
    model: Optional[object] = None
    insert_list: List = field(default_factory=list)
    a: float = 0.0
    b: float = 0.0
    err_min: float = 0.0
    err_max: float = 0.0
    dict_circle: List = field(default_factory=list)

    def set_main_ref_dis_arr(self, dis):
        self.dis = dis

    def set_i_value_pts(self, iValuePts):
        self.iValuePts = iValuePts

    def set_linear(self, a, b, err_min, err_max):
        self.a = a
        self.b = b
        self.err_min = err_min
        self.err_max = err_max

    def set_dict(self, dict_circle):
        self.dict_circle = dict_circle

    def set_coeffs(self, coeffs):
        self.coeffs = coeffs

    def set_model(self, model):
        self.model = model
        
    def set_insert_pt(self, insert_list):
        self.insert_list = insert_list

@dataclass
class Ref_Set:
    ref_set: List[Ref_Point] = field(default_factory=list)
