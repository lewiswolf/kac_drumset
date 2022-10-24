# core
from unittest import TestCase

# dependencies
import cv2					# image processing
import numpy as np 			# maths

# src
from kac_drumset.geometry import (
	booleanMask,
	convexNormalisation,
	isColinear,
	isConvex,
	largestVector,
	RandomPolygon,
	Polygon,
)


class GeometryTests(TestCase):
	'''
	Tests used in conjunction with `/geometry`.
	'''

	def test_polygon(self) -> None:
		'''
		Test properties of the type Polygon.
		'''

		squares = [
			Polygon(np.array([[0, 0], [1, 0], [1, 1], [0, 1]])),
			Polygon(np.array([[0, 0], [0, 1], [1, 1], [1, 0]])),
		]

		for polygon in squares:
			# This test asserts that a polygon has the correct number of vertices.
			self.assertEqual(len(polygon.vertices), polygon.n)

			# This test asserts that isConvex() works for any closed arrangement of vertices.
			self.assertTrue(isConvex(polygon.vertices))

		# This test asserts that after convexNormalisation, one of the above squares produces the correct output.
		self.assertFalse(False in np.equal(
			convexNormalisation(squares[0].vertices),
			np.array([
				[0., 0.5],
				[0.5, 1.],
				[1., 0.5],
				[0.5, 0.],
			]),
		))

		# This test asserts that after convexNormalisation, the two squares produce the same output.
		# The two squares have opposite vertex order.
		self.assertFalse(False in np.equal(
			convexNormalisation(squares[0].vertices),
			convexNormalisation(squares[1].vertices),
		))

		# This test asserts that after convexNormalisation, the quads produce the same output.
		# The first two quads have opposite vertex order.
		# The second two quads have their x and y coordinates swapped.
		quads = [
			Polygon(np.array([[0, 0], [1.1, 0], [1, 1], [0, 1]])),
			Polygon(np.array([[0, 0], [0, 1], [1, 1], [1.1, 0]])),
			Polygon(np.array([[0, 0], [0, 1.1], [1, 1], [1, 0]])),
			Polygon(np.array([[0, 0], [1, 0], [1, 1], [0, 1.1]])),
		]
		for quad in quads:
			quad.vertices = convexNormalisation(quad.vertices)
		self.assertFalse(False in np.equal(quads[0].vertices, quads[1].vertices))
		# np.allclose is used, as opposed to np.equal, to account for floating point errors.
		self.assertTrue(np.allclose(quads[0].vertices, quads[2].vertices))
		self.assertTrue(np.allclose(quads[0].vertices, quads[3].vertices))

	def test_random_polygon(self) -> None:
		'''
		Stress test multiple properties of the class RandomPolygon.
		'''

		for i in range(10000):
			polygon = RandomPolygon(20, allow_concave=True)
			LV = largestVector(polygon.vertices)

			# This test asserts that a polygon has the correct number of vertices.
			self.assertEqual(len(polygon.vertices), polygon.n)

			# This test asserts that the vertices are strictly bounded between 0.0 and 1.0.
			self.assertEqual(np.min(polygon.vertices), 0.)
			self.assertEqual(np.max(polygon.vertices), 1.)

			# This test asserts that the largest vector is of magnitude 1.0.
			self.assertEqual(LV[0], 1.0)

			# This test asserts that the area(), used for calculating the area of a polygon is accurate to at least 6 decimal
			# places. This comparison is bounded due to the area() being 64-bit, whilst the comparison function,
			# cv2.contourArea(), is 32-bit.
			self.assertAlmostEqual(
				polygon.area,
				cv2.contourArea(polygon.vertices.astype('float32')),
				places=7,
			)

			# This test asserts that no 3 adjacent vertices are colinear.
			for j in range(polygon.n):
				self.assertFalse(isColinear(np.array([
					polygon.vertices[j - 1 if j > 0 else polygon.n - 1],
					polygon.vertices[j],
					polygon.vertices[(j + 1) % polygon.n],
				])))

			if polygon.convex:
				# This test asserts that all supposedly convex polygons are in fact convex. As a result, if this test passes, we
				# can assume that the generateConvexPolygon() function works as intended.
				self.assertTrue(isConvex(polygon.vertices))

				# This test asserts that the largest vector lies across the x-axis.
				self.assertTrue(polygon.vertices[LV[1][0]][0] == 0.)
				self.assertTrue(polygon.vertices[LV[1][1]][0] == 1.)

				# This test asserts that the calculated centroid lies within the polygon. For concave shapes, this test may fail.
				mask = booleanMask(polygon.vertices, 100, convex=polygon.convex)
				self.assertEqual(mask[
					round(polygon.centroid[0] * 100),
					round(polygon.centroid[1] * 100),
				], 1)

				# This test asserts that convexNormalisation does not continuously alter the polygon.
				# np.allclose is used, as opposed to np.equal, to account for floating point errors.
				self.assertTrue(np.allclose(polygon.vertices, convexNormalisation(polygon.vertices)))
