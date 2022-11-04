'''
Functions for generating polygons.
'''

# dependencies
import numpy as np 			# maths
import numpy.typing as npt	# typing for numpy

__all__ = ['generateConcavePolygon']


# TO FIX: see todo.md => `Missing a reliable algorithm to generate all concave shapes`
def generateConcavePolygon(N: int) -> npt.NDArray[np.float64]:
	'''
	Generates a random concave polygon, with a small probability of also returning a convex shape.
	It should be noted that this function cannot be used to create all possible simple polygons;
	see todo.md => 'Missing a reliable algorithm to generate all concave shapes'.
	'''

	vertices = np.random.random((N, 2))
	# center around the origin
	vertices[:, 0] -= (np.max(vertices[:, 0]) + np.min(vertices[:, 0])) / 2
	vertices[:, 1] -= (np.max(vertices[:, 1]) + np.min(vertices[:, 1])) / 2
	# order by polar angle theta
	vertices = vertices[np.argsort(np.arctan2(vertices[:, 1], vertices[:, 0]))]
	return vertices
