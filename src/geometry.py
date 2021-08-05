'''
This file contains functions relating to computational geometry, such as generating
random polygons, and converting them into bit masks. All of these functions are
controlled by the class RandomPoly().
'''

# core
import random

# dependencies
import cv2
import numpy as np 						# maths
import numpy.typing as npt				# typing for numpy

# src
from settings import PhysicalModelSettings, settings
pmSettings: PhysicalModelSettings = settings['PM_SETTINGS']


class RandomPolygon():
	'''
	This class is used to generate a random polygon, normalised and centered between 0.0
	and 1.0. The resultant vertices are then projected onto a discrete matrix.
	'''

	grid: npt.NDArray[np.int8]			# discrete projection of the polygon, where 1 signifies a bounded region of the shape
	n: int								# the number of vertices
	vertices: npt.NDArray[np.float64]	# cartesian products representing the corners of a shape

	def __init__(self) -> None:
		self.n = random.randint(3, pmSettings['max_vertices'])
		self.grid = np.zeros((pmSettings['grid_size'], pmSettings['grid_size']), 'int8')

		if not pmSettings['allow_concave'] or random.getrandbits(1):
			self.vertices = generateConvex(self.n)
			cv2.fillConvexPoly(
				self.grid,
				np.array([[round(x * (pmSettings['grid_size'] - 1)), round(y * (pmSettings['grid_size'] - 1))] for [x, y] in self.vertices], 'int32'),
				1,
			)
		else:
			self.vertices = generateConcave(self.n)
			cv2.fillPoly(
				self.grid,
				np.array([[[round(x * (pmSettings['grid_size'] - 1)), round(y * (pmSettings['grid_size'] - 1))] for [x, y] in self.vertices]], 'int32'),
				1,
			)


def generateConcave(n: int) -> npt.NDArray[np.float64]:
	'''
	Generate a random concave shape, with a small probability of also returning a
	convex shape.
	'''

	# TO ADD: see todo.md -> 'Missing a reliable algorithm to generate all concave shapes'
	vertices = np.random.random((n, 2))
	# center around the origin
	x_min = np.min(vertices[:, 0])
	x_max = np.max(vertices[:, 0])
	y_min = np.min(vertices[:, 1])
	y_max = np.max(vertices[:, 1])
	vertices[:, 0] -= (x_max + x_min) / 2
	vertices[:, 1] -= (y_max + y_min) / 2
	# order by polar angle theta
	vertices = vertices[np.argsort(np.arctan2(vertices[:, 1], vertices[:, 0]))]
	# normalise and centre polygon on positive quadrant
	vertices *= 1 / max((x_max - x_min), (y_max - y_min))
	vertices += 0.5
	# correct floating point error
	if np.min(vertices) != 0.0:
		i = np.argmin(vertices)
		vertices[int(np.floor(i / 2)), i % 2] = 0.0
	return vertices


def generateConvex(n: int) -> npt.NDArray[np.float64]:
	'''
	Generate convex shappes according to Pavel Valtr's 1995 alogrithm. Adapted from
	Sander Verdonschot's Java version, found here:
		https://cglab.ca/~sander/misc/ConvexGeneration/ValtrAlgorithm.java
	'''

	# initialise random coordinates
	X_rand = np.sort(np.random.random(n))
	Y_rand = np.sort(np.random.random(n))
	X_new = np.zeros(n)
	Y_new = np.zeros(n)

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
	x_min = np.min(vertices[:, 0])
	x_max = np.max(vertices[:, 0])
	y_min = np.min(vertices[:, 1])
	y_max = np.max(vertices[:, 1])
	vertices[:, 0] += ((x_max - x_min) / 2) - x_max
	vertices[:, 1] += ((y_max - y_min) / 2) - y_max

	# normalise and centre polygon on positive quadrant
	vertices *= 1 / max((x_max - x_min), (y_max - y_min))
	vertices += 0.5
	# correct floating point error
	if np.min(vertices) != 0.0:
		j = np.argmin(vertices)
		vertices[int(np.floor(j / 2)), j % 2] = 0.0
	return vertices
