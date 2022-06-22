'''
This file contains various functions relating to computational geometry. These include:
	-	Calculating the area of the polygon.
	-	Creating a discrete matrix representation of the polygon. (boolean mask)
	- 	Calculating the centroid of a polygon.
	-	Generating random polygons, which can either be convex or concave.
	-	Normalisation based on group theory transformations.
	-	Determine whether or a not a set of three vertices are colinear.
	- 	Determine whether or not a given polygon is convex.
	- 	Calculating the largest vector within a polygon.
'''

# core
from typing import Optional

# dependencies
import cv2					# image processing
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

# src
from . import _geometry as _G

__all__ = [
	'area',
	'booleanMask',
	'centroid',
	'generateConcave',
	'groupNormalisation',
	'largestVector',
]


def area(vertices: npt.NDArray[np.float64]) -> float:
	'''
	An implementation of the shoelace algorithm, first described by Albrecht Ludwig
	Friedrich Meister, which is used to calculate the area of a polygon. The area
	of a polygon can also be computed (using Green's theorem directly) using
	`cv2.contourArea(self.vertices.astype('float32'))`. However, this function
	requires that the input be of the type float32, resulting in a trade off between
	(marginal) performance gains and lower precision.
	'''

	n = vertices.shape[0]
	return abs(sum([
		vertices[i, 0] * vertices[(i + 1) % n, 1]
		- vertices[i, 1] * vertices[(i + 1) % n, 0]
		for i in range(n)
	])) / 2


def booleanMask(
	vertices: npt.NDArray[np.float64],
	grid_size: int,
	convex: Optional[bool],
) -> npt.NDArray[np.int8]:
	'''
	This function creates a boolean mask of an input polygon on a grid with dimensions
	R^(grid_size). The input shape should exist within a domain R^G where G ∈ [0, 1].
	'''

	if convex is None:
		convex = _G.isConvex(vertices)

	mask = np.zeros((grid_size, grid_size), 'int8')
	if convex:
		cv2.fillConvexPoly(
			mask,
			np.array([[
				round(x * (grid_size - 1)),
				round(y * (grid_size - 1)),
			] for [x, y] in vertices], 'int32'),
			1,
		)
	else:
		cv2.fillPoly(
			mask,
			np.array([[[
				round(x * (grid_size - 1)),
				round(y * (grid_size - 1)),
			] for [x, y] in vertices]], 'int32'),
			1,
		)
	return np.transpose(mask) # this maintains that mask[x, y] works as intended


def centroid(vertices: npt.NDArray[np.float64], area: float) -> tuple[float, float]:
	'''
	This algorithm is used to calculate the geometric centroid of a 2D polygon.
	See http://paulbourke.net/geometry/polygonmesh/ 'Calculating the area and
	centroid of a polygon'.
	'''

	n = vertices.shape[0]
	if n == 3:
		# Triangles have a much simpler formula, and so these are caluclated separately.
		return (sum(vertices[:, 0]) / 3, sum(vertices[:, 1]) / 3)

	return (
		abs(sum([
			(vertices[i, 0] + vertices[(i + 1) % n, 0])
			* (vertices[i, 0] * vertices[(i + 1) % n, 1] - vertices[(i + 1) % n, 0] * vertices[i, 1])
			for i in range(n)
		]) / (6 * area)),
		abs(sum([
			(vertices[i, 1] + vertices[(i + 1) % n, 1])
			* (vertices[i, 0] * vertices[(i + 1) % n, 1] - vertices[(i + 1) % n, 0] * vertices[i, 1])
			for i in range(n)
		]) / (6 * area)),
	)


# TO FIX: see todo.md => `Missing a reliable algorithm to generate all concave shapes`
def generateConcave(n: int) -> npt.NDArray[np.float64]:
	'''
	Generates a random concave shape, with a small probability of also returning a
	convex shape. It should be noted that this function can not be used to create
	all possible simple polygons; see todo.md => 'Missing a reliable algorithm to
	generate all concave shapes'.
	'''

	vertices = np.random.random((n, 2))
	# center around the origin
	vertices[:, 0] -= (np.max(vertices[:, 0]) + np.min(vertices[:, 0])) / 2
	vertices[:, 1] -= (np.max(vertices[:, 1]) + np.min(vertices[:, 1])) / 2
	# order by polar angle theta
	vertices = vertices[np.argsort(np.arctan2(vertices[:, 1], vertices[:, 0]))]
	return vertices


# TO FIX: see todo.md => `groupNormalisation`
def groupNormalisation(
	vertices: npt.NDArray[np.float64],
	convex: Optional[bool],
) -> npt.NDArray[np.float64]:
	'''
	This function uses the largest vector to define a polygon's span across the
	y-axis. After finding the largest vector, the polygon is rotated about said
	vector's midpoint, and finally the entire polygon is normalised to span the
	unit interval.
	'''

	if convex is None:
		convex = _G.isConvex(vertices)

	# rotate around the midpoint of the largest vector
	_, idx = largestVector(vertices)
	vertices[:, 0] -= (vertices[idx[0], 0] + vertices[idx[1], 0]) / 2
	vertices[:, 1] -= (vertices[idx[0], 1] + vertices[idx[1], 1]) / 2
	theta = 0.0 - np.arctan(vertices[idx[0], 0] / vertices[idx[0], 1])
	cos_theta, sin_theta = np.cos(theta), np.sin(theta)
	for i, (x, y) in enumerate(vertices):
		vertices[i, 0] = (x * cos_theta + y * sin_theta)
		vertices[i, 1] = (-x * sin_theta + y * cos_theta)

	if convex:
		pass
	else:
		pass

	# normalise to unit interval
	vertices[:, 0] -= (np.min(vertices[:, 0]) + np.max(vertices[:, 0])) / 2
	v_min = np.min(vertices)
	vertices = (vertices - v_min) / (np.max(vertices) - v_min)
	return vertices


def largestVector(vertices: npt.NDArray[np.float64]) -> tuple[float, tuple[int, int]]:
	'''
	This function tests each pair of vertices in a given polygon to find the largest
	vector, and returns the length of the vector and its indices.
	'''

	n = vertices.shape[0]
	vec_max = 0.0
	idx = (0, 0)
	for i in range(n):
		for j in range(i + 1, n):
			vec = ((vertices[i, 0] - vertices[j, 0]) ** 2 + (vertices[i, 1] - vertices[j, 1]) ** 2) ** 0.5
			if vec > vec_max:
				vec_max = vec
				idx = (i, j)
	return vec_max, idx
