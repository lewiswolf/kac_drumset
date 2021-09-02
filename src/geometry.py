'''
This file contains various functions relating to computational geometry. These include:
	-	Generating random polygons, which can either be convex or concave.
	-	Creating a discrete matrix representation of the polygon. (boolean mask)
	-	Calulating the area of the polygon, as well as its centroid.
'''

# core
import random
import math

# dependencies
import cv2								# image processing
import numpy as np 						# maths
import numpy.typing as npt				# typing for numpy


class RandomPolygon():
	'''
	This class is used to generate a random polygon, normalised and centered between 0.0
	and 1.0. Various properties relating to this polygon are also attached to this class.
	'''

	area: float							# area of the polygon
	centroid: tuple[float, float]		# coordinate pair representing the centroid of the polygon
	convex: bool						# is the polygon convex?
	mask: npt.NDArray[np.int8]			# discrete projection of the polygon, where 1 signifies the internal region of the shape
	n: int								# the number of vertices
	vertices: npt.NDArray[np.float64]	# cartesian products representing the corners of a shape

	def __init__(self, maxVertices: int, gridSize: int, allowConcave: bool = True) -> None:
		'''
		This function generates a polygon, whilst also calculating its properties.
		params:
			maxVertices:	Maximum amount of vertices. The true value is a uniform
							distribution from 3 to maxVertices.
			gridSize:		Every polygon has a boolean mask as one of its properties.
							This variable controls the size of that mask.
			allowConcave:	Is this polygon allowed to be concave?
		'''

		# generate random polygon
		self.n = random.randint(3, maxVertices)
		if not allowConcave or random.getrandbits(1):
			self.vertices = generateConvex(self.n)
			self.convex = True
		else:
			self.vertices = generateConcave(self.n)
			self.convex = isConvex(self.n, self.vertices)

		# calculate other properties
		self.area = shoelaceFormula(self.n, self.vertices)
		self.centroid = locateCentroid(self.area, self.n, self.vertices)

		# compute the boolean mask
		self.mask = np.zeros((gridSize, gridSize), 'int8')
		if self.convex:
			cv2.fillConvexPoly(
				self.mask,
				np.array([[
					round(x * (gridSize - 1)),
					round(y * (gridSize - 1)),
				] for [x, y] in self.vertices], 'int32'),
				1,
			)
		else:
			cv2.fillPoly(
				self.mask,
				np.array([[[
					round(x * (gridSize - 1)),
					round(y * (gridSize - 1)),
				] for [x, y] in self.vertices]], 'int32'),
				1,
			)
		self.mask = np.transpose(self.mask) # this maintains that mask[x, y] works as intended


def locateCentroid(area: float, n: int, vertices: npt.NDArray[np.float64]) -> tuple[float, float]:
	'''
	The below algorithm is used to calculate the geometric centroid of a 2D polygon.
	See http://paulbourke.net/geometry/polygonmesh/ 'Calculating the area and
	centroid of a polygon'.
	'''

	if n == 3:
		# Triangles have a much simpler formula, and so these are caluclated seperately.
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


def generateConcave(n: int) -> npt.NDArray[np.float64]:
	'''
	Generates a random concave shape, with a small probability of also returning a
	convex shape. It should be noted that this function can not be used to create
	all possible simple polygons; see todo.md => 'Missing a reliable algorithm to
	generate all concave shapes'.
	'''

	vertices = np.random.random((n, 2))
	# center around the origin
	x_min, x_max = np.min(vertices[:, 0]), np.max(vertices[:, 0])
	y_min, y_max = np.min(vertices[:, 1]), np.max(vertices[:, 1])
	vertices[:, 0] -= (x_max + x_min) / 2
	vertices[:, 1] -= (y_max + y_min) / 2
	# order by polar angle theta
	vertices = vertices[np.argsort(np.arctan2(vertices[:, 1], vertices[:, 0]))]
	# normalise and centre polygon on positive quadrant
	vertices *= 1 / max((x_max - x_min), (y_max - y_min))
	vertices += 0.5
	correctFloatingPointError(vertices)
	return vertices


def generateConvex(n: int) -> npt.NDArray[np.float64]:
	'''
	Generate convex shappes according to Pavel Valtr's 1995 alogrithm. Adapted from
	Sander Verdonschot's Java version, found here:
		https://cglab.ca/~sander/misc/ConvexGeneration/ValtrAlgorithm.java
	'''

	# initialise random coordinates
	X_rand, Y_rand = np.sort(np.random.random(n)), np.sort(np.random.random(n))
	X_new, Y_new = np.zeros(n), np.zeros(n)

	# divide the interior points into two chains
	lastTop = lastBot = X_rand[0]
	lastLeft = lastRight = Y_rand[0]
	for i in range(1, n - 1):
		if random.getrandbits(1):
			X_new[i] = X_rand[i] - lastTop
			lastTop = X_rand[i]
			Y_new[i] = Y_rand[i] - lastLeft
			lastLeft = Y_rand[i]
		else:
			X_new[i] = lastBot - X_rand[i]
			lastBot = X_rand[i]
			Y_new[i] = lastRight - Y_rand[i]
			lastRight = Y_rand[i]
	X_new[0] = X_rand[n - 1] - lastTop
	X_new[n - 1] = lastBot - X_rand[n - 1]
	Y_new[0] = Y_rand[n - 1] - lastLeft
	Y_new[n - 1] = lastRight - Y_rand[n - 1]

	# randomly combine x and y and sort by polar angle
	np.random.shuffle(Y_new)
	vertices = np.stack((X_new, Y_new), axis=-1)
	vertices = vertices[np.argsort(np.arctan2(vertices[:, 1], vertices[:, 0]))]

	# arrange to points end to end to form a polygon
	x_accum = y_accum = 0
	for i, [x, y] in enumerate(vertices):
		vertices[i] = [x_accum, y_accum]
		x_accum += x
		y_accum += y

	# center around the origin
	x_min, x_max = np.min(vertices[:, 0]), np.max(vertices[:, 0])
	y_min, y_max = np.min(vertices[:, 1]), np.max(vertices[:, 1])
	vertices[:, 0] += ((x_max - x_min) / 2) - x_max
	vertices[:, 1] += ((y_max - y_min) / 2) - y_max

	# normalise and centre polygon on positive quadrant
	vertices *= 1 / max((x_max - x_min), (y_max - y_min))
	vertices += 0.5
	correctFloatingPointError(vertices)
	return vertices


def correctFloatingPointError(vertices: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
	'''
	Corrects a floating point error which occurs when normilising the vertices
	between 0.0 and 1.0. See `unit_tests.py` => `test_floating_point_error()`.
	'''

	try:
		assert np.min(vertices) == 0.0
		assert np.max(vertices) == 1.0
	except AssertionError:
		i = np.argmin(vertices)
		j = np.argmax(vertices)
		vertices[math.floor(i / 2), i % 2] = 0.0
		vertices[math.floor(j / 2), j % 2] = 1.0
	return vertices


def shoelaceFormula(n: int, vertices: npt.NDArray[np.float64]) -> float:
	'''
	An implementation of the shoelace algorithm, first described by Albrecht Ludwig
	Friedrich Meister, which is used to calculate the area of a polygon. The area
	of a polygon can also be computed (using Green's theorem directly) using
	`cv2.contourArea(self.vertices.astype('float32'))`. However, this function requires
	that the input be of the type float32, resulting in a trade off between (marginal)
	performance gains and lower precision.
	'''

	return abs(sum([
		vertices[i, 0] * vertices[(i + 1) % n, 1]
		- vertices[i, 1] * vertices[(i + 1) % n, 0]
		for i in range(n)
	])) / 2


def isConvex(n: int, vertices: npt.NDArray[np.float64]) -> bool:
	'''
	Tests whether or not a given array of vertices forms a convex polygon. This is
	achieved using the resultant sign of the cross product for each vertex:
		[(x_i - x_i-1), (y_i - y_i-1)] x [(x_i+1 - x_i), (y_i+1 - y_i)]
	See => http://paulbourke.net/geometry/polygonmesh/ 'Determining whether or not a
	polygon (2D) has its vertices ordered clockwise or counter-clockwise'.
	'''

	for b in range(n):
		a = b - 1 if b > 0 else n - 1
		c = b + 1 if b < n - 1 else 0
		if np.cross(
			np.array([vertices[b, 0] - vertices[a, 0], vertices[b, 1] - vertices[a, 1]]),
			np.array([vertices[c, 0] - vertices[b, 0], vertices[c, 1] - vertices[b, 1]]),
		) < 0:
			return False
	return True
