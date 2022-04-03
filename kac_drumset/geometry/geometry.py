'''
This file contains various functions relating to computational geometry. These include:
	-	Calculating the area of the polygon.
	-	Creating a discrete matrix representation of the polygon. (boolean mask)
	- 	Calculating the centroid of a polygon.
	-	Generating random polygons, which can either be convex or concave.
	-	Normalisation based on group theory transformations.
	-	Determing whether or a not a set of three vertices are colinear.
	- 	Determing whether or not a given polygon is convex.
	- 	Calculating the largest vector within a polygon.
'''

# core
import random
from typing import Optional

# dependencies
import cv2					# image processing
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

__all__ = [
	'area',
	'booleanMask',
	'centroid',
	'generateConcave',
	'generateConvex',
	'groupNormalisation',
	'isColinear',
	'isConvex',
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
		convex = isConvex(vertices)

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


def generateConvex(n: int) -> npt.NDArray[np.float64]:
	'''
	Generate convex shapes according to Pavel Valtr's 1995 algorithm. Adapted
	from Sander Verdonschot's Java version, found here:
		https://cglab.ca/~sander/misc/ConvexGeneration/ValtrAlgorithm.java
	'''

	# initialise random coordinates
	X, Y = np.zeros(n), np.zeros(n)
	X_rand, Y_rand = np.sort(np.random.random(n)), np.sort(np.random.random(n))

	# divide the interior points into two chains
	last_true, last_false = 0, 0
	for i in range(1, n):
		if i != n - 1:
			if random.getrandbits(1):
				X[i] = X_rand[i] - X_rand[last_true]
				Y[i] = Y_rand[i] - Y_rand[last_true]
				last_true = i
			else:
				X[i] = X_rand[last_false] - X_rand[i]
				Y[i] = Y_rand[last_false] - Y_rand[i]
				last_false = i
		else:
			X[0] = X_rand[i] - X_rand[last_true]
			Y[0] = Y_rand[i] - Y_rand[last_true]
			X[i] = X_rand[last_false] - X_rand[i]
			Y[i] = Y_rand[last_false] - Y_rand[i]

	# randomly combine x and y and sort by polar angle
	np.random.shuffle(Y)
	vertices = np.stack((X, Y), axis=-1)
	vertices = vertices[np.argsort(np.arctan2(vertices[:, 1], vertices[:, 0]))]

	# arrange points end to end to form a polygon
	vertices = np.cumsum(vertices, axis=0)

	# center around the origin
	x_max, y_max = np.max(vertices[:, 0]), np.max(vertices[:, 1])
	vertices[:, 0] += ((x_max - np.min(vertices[:, 0])) / 2) - x_max
	vertices[:, 1] += ((y_max - np.min(vertices[:, 1])) / 2) - y_max
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
		convex = isConvex(vertices)

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


def isColinear(vertices: npt.NDArray[np.float64]) -> bool:
	'''
	Determines whether or not a given set of three vertices are colinear.
	'''

	if vertices.shape != (3, 2):
		raise ValueError('isColinear() only supports an input of vertices with shape (3, 2).')
	return (
		(vertices[2, 1] - vertices[1, 1]) * (vertices[1, 0] - vertices[0, 0])
		== (vertices[1, 1] - vertices[0, 1]) * (vertices[2, 0] - vertices[1, 0])
	)


def isConvex(vertices: npt.NDArray[np.float64]) -> bool:
	'''
	Tests whether or not a given array of vertices forms a convex polygon. This is
	achieved using the resultant sign of the cross product for each vertex:
		[(x_i - x_i-1), (y_i - y_i-1)] x [(x_i+1 - x_i), (y_i+1 - y_i)]
	See => http://paulbourke.net/geometry/polygonmesh/ 'Determining whether or not
	a polygon (2D) has its vertices ordered clockwise or counter-clockwise'.
	'''

	n: int = vertices.shape[0]
	convex: Optional[bool] = None
	for i in range(n):
		# ascertain if the polygon is ordered clockwise or counter-clockwise
		if convex is None:
			convex = np.cross(
				vertices[i] - vertices[i - 1 if i > 0 else n - 1],
				vertices[(i + 1) % n] - vertices[i],
			) < 0
			continue
		# assert that all other vertices are the same
		if (np.cross(
			vertices[i] - vertices[i - 1 if i > 0 else n - 1],
			vertices[(i + 1) % n] - vertices[i],
		) < 0) != convex:
			return False
	return True


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