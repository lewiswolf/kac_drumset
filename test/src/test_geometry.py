# core
from unittest import TestCase

# dependencies
import cv2					# image processing
import numpy as np 			# maths

# src
from kac_drumset.geometry import Polygon, RandomPolygon, booleanMask, isColinear, isConvex, largestVector


class GeometryTests(TestCase):
	'''
	Tests used in conjunction with `geometry.py` and types Polygon and RandomPolygon.
	'''

	def test_polygon(self) -> None:
		'''
		Test properties of the type Polygon.
		'''

		for polygon in [
			Polygon(np.array([[0, 0], [1, 0], [1, 1], [0, 1]])),
			Polygon(np.array([[0, 0], [0, 1], [1, 1], [1, 0]])),
		]:
			# This test asserts that a polygon has the correct number of vertices.
			self.assertEqual(len(polygon.vertices), polygon.n)

			# This test asserts that isConvex() works for any closed arrangement of vertices.
			self.assertTrue(isConvex(polygon.vertices))

	def test_random_polygon(self) -> None:
		'''
		Stress test multiple properties of the class RandomPolygon.
		'''

		for i in range(10000):
			polygon = RandomPolygon(20, allow_concave=True)

			# This test asserts that a polygon has the correct number of vertices.
			self.assertEqual(len(polygon.vertices), polygon.n)

			# This test asserts that the vertices are strictly bounded between 0.0 and 1.0.
			self.assertEqual(np.min(polygon.vertices), 0.0)
			self.assertEqual(np.max(polygon.vertices), 1.0)

			# This test asserts that the largest vector is of magnitude 1.0.
			self.assertEqual(largestVector(polygon.vertices)[0], 1.0)

			# This test asserts that the area(), used for calculating the area of a polygon is accurate to at least 6 decimal
			# places. This comparison is bounded due to the area() being 64-bit, whilst the comparison function,
			# cv2.contourArea(), is 32-bit.
			self.assertAlmostEqual(
				polygon.area,
				cv2.contourArea(polygon.vertices.astype('float32')),
				places=6,
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
				# can assume that the generateConvex() function works as intended.
				self.assertTrue(isConvex(polygon.vertices))

				# This test asserts that the calculated centroid lies within the polygon. For concave shapes, this test may fail.
				mask = booleanMask(polygon.vertices, 100, convex=polygon.convex)
				self.assertEqual(mask[
					round(polygon.centroid[0] * 100),
					round(polygon.centroid[1] * 100),
				], 1)
