'''
'''

# core
import random

# dependencies
import numpy as np 				# maths
import numpy.typing as npt		# typing for numpy


ALLOW_CONCAVE = True			# are concave shapes allowed?
MAX_VERTICES = 5				# maximum amount of vertices for a given shape


class RandomPolygon():
	'''
	This class is used to generate a random shape, and discretise its vertices
	onto a uniform grid.
	'''

	n: int									# the number of vertices
	vertices: npt.NDArray[np.float64]		# cartesian products representing the corners of a shape

	def __init__(self) -> None:
		self.n = random.randint(3, MAX_VERTICES)

		if not ALLOW_CONCAVE:
			self.vertices = generateConvex(self.n)
		elif random.getrandbits(1):
			self.vertices = generateConcave(self.n)
		else:
			self.vertices = generateConvex(self.n)


def generateConcave(n: int) -> npt.NDArray[np.float64]:
	'''
	Generate a random concave shape, with a small probability of also returning a
	convex shape.
	'''

	# TO ADD: see todo.md -> 'Missing a reliable algorithm to generate all concave shapes'
	vertices = np.random.random((n, 2))
	# center around the origin
	vertices[:, 0] -= (np.max(vertices[:, 0]) + np.min(vertices[:, 0])) / 2
	vertices[:, 1] -= (np.max(vertices[:, 1]) + np.min(vertices[:, 1])) / 2
	# order by polar angle theta
	vertices = vertices[np.argsort(np.arctan2(vertices[:, 1], vertices[:, 0]))]
	# centre polygon on positive quadrant
	vertices += 0.5
	return vertices


def generateConvex(n: int) -> npt.NDArray[np.float64]:
	'''
	Generate convex shappes according to Pavel Valtr's 1995 alogrithm. Ported from
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

	# center the polygon between 0 and 1
	x_min = np.min(vertices[:, 0])
	y_min = np.min(vertices[:, 1])
	vertices[:, 0] += ((1 - (np.max(vertices[:, 0]) - x_min)) / 2) - x_min
	vertices[:, 1] += ((1 - (np.max(vertices[:, 1]) - y_min)) / 2) - y_min

	return vertices
