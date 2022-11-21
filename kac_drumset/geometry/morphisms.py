'''
Functions for producing group theoretic transformations.
'''

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from ._polygon_properties import largestVector
from .types import Polygon

__all__ = ['concaveNormalisation']


# TO FIX: see todo.md => `concaveNormalisation`
def concaveNormalisation(P: Polygon) -> npt.NDArray[np.float64]:
	'''
	This function uses the largest vector to define a polygon's span across the y-axis. After finding the largest vector,
	the polygon is rotated about said vector's midpoint, and finally the entire polygon is normalised to span the unit
	interval.
	'''

	# rotate around the midpoint of the largest vector
	vertices = P.vertices
	_, idx = largestVector(P)
	vertices[:, 0] -= (vertices[idx[0], 0] + vertices[idx[1], 0]) / 2
	vertices[:, 1] -= (vertices[idx[0], 1] + vertices[idx[1], 1]) / 2
	theta = 0. - np.arctan(vertices[idx[0], 0] / vertices[idx[0], 1])
	cos_theta, sin_theta = np.cos(theta), np.sin(theta)
	for i, (x, y) in enumerate(vertices):
		vertices[i, 0] = (x * cos_theta + y * sin_theta)
		vertices[i, 1] = (-x * sin_theta + y * cos_theta)

	# normalise to unit interval
	vertices[:, 0] -= (np.min(vertices[:, 0]) + np.max(vertices[:, 0])) / 2
	v_min = np.min(vertices)
	vertices = (vertices - v_min) / (np.max(vertices) - v_min)
	return vertices
