'''
Methods to derive spectral geometry properties.
'''

# src
from .types import Shape

__all__ = ['weylCondition']


def weylCondition(S_1: Shape, S_2: Shape) -> bool:
	'''
	Using Weyl's asymptotic law, determine whether two polygons may be isospectral.
	https://en.wikipedia.org/wiki/Weyl_law
	'''
	return S_1.area() == S_2.area()
