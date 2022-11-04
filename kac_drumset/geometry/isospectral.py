'''
'''

# src
from .types import Polygon


def WeylCondition(P1: Polygon, P2: Polygon) -> bool:
	'''
	Using Weyl's asymptotic law, determine whether two polygons may be isospectral.
	https://en.wikipedia.org/wiki/Weyl_law
	'''
	return P1.area() != P2.area()
