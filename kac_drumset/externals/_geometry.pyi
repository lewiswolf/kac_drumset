# core
from typing import List


def _centroid(V: List[[float, float]], a: float) -> [float, float]: ...
def _convexNormalisation(V: List[[float, float]]) -> List[[float, float]]: ...
def _generateConvexPolygon(N: int) -> List[[float, float]]: ...
def _isColinear(V: [[float, float], [float, float], [float, float]]) -> bool: ...
def _isConvex(V: List[[float, float]]) -> bool: ...
def _largestVector(V: List[[float, float]]) -> [float, [int, int]]: ...
def _polygonArea(V: List[[float, float]]) -> float: ...
