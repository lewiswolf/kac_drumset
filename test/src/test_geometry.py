# core
from unittest import TestCase

# dependencies
import cv2					# image processing
import numpy as np 			# maths

# src
from kac_drumset.geometry import (
	centroid,
	convexNormalisation,
	drawCircle,
	drawPolygon,
	generateConvexPolygon,
	isColinear,
	isPointInsidePolygon,
	largestVector,
	RandomPolygon,
	UnitRectangle,
	UnitTriangle,
	Circle,
	Polygon,
)


class GeometryTests(TestCase):
	'''
	Tests used in conjunction with `/geometry`.
	'''

	def test_circle(self) -> None:
		'''
		Test properties of the type Circle.
		'''

		# This test asserts that the center of the boolean mask is always true.
		for r in [0.1, 0.25, 0.5, 1.]:
			C = Circle(r)
			M = drawCircle(C, 101)
			self.assertEqual(M[50, 50], 1)
			self.assertEqual(M[0, 0], 0)
			self.assertEqual(M[0, 100], 0)
			self.assertEqual(M[100, 0], 0)
			self.assertEqual(M[100, 100], 0)

	def test_convex_polygon(self) -> None:
		'''
		Test properties of the type Polygon.
		'''

		# Two squares ordered clockwise and counter-clockwise respectively.
		squares = [
			Polygon([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
			Polygon([[0., 0.], [0., 1.], [1., 1.], [1., 0.]]),
		]
		# The first two quads have opposite vertex order.
		# The second two quads have their x and y coordinates swapped.
		quads = [
			Polygon([[0, 0], [1.1, 0], [1, 1], [0, 1]]),
			Polygon([[0, 0], [0, 1], [1, 1], [1.1, 0]]),
			Polygon([[0, 0], [0, 1.1], [1, 1], [1, 0]]),
			Polygon([[0, 0], [1, 0], [1, 1], [0, 1.1]]),
		]

		# This test asserts that isPointInsidePolygon works as expected
		for quad in quads:
			for p in quad.vertices:
				self.assertTrue(isPointInsidePolygon(p, quad))
			for n in range(quad.N):
				a = quad.vertices[n]
				b = quad.vertices[(n + 1) % quad.N]
				self.assertTrue(isPointInsidePolygon(((a[0] + b[0]) / 2., (a[1] + b[1]) / 2.), quad))
		for square in squares:
			for p in square.vertices:
				self.assertTrue(isPointInsidePolygon(p, square))
			for n in range(square.N):
				a = square.vertices[n]
				b = square.vertices[(n + 1) % square.N]
				self.assertTrue(isPointInsidePolygon(((a[0] + b[0]) / 2., (a[1] + b[1]) / 2.), square))
			self.assertTrue(isPointInsidePolygon((0.999, 0.5), square))
			self.assertFalse(isPointInsidePolygon((1.001, 0.5), square))
			self.assertTrue(isPointInsidePolygon((0.5, 0.999), square))
			self.assertFalse(isPointInsidePolygon((0.5, 1.001), square))
			self.assertTrue(isPointInsidePolygon((0.001, 0.5), square))
			self.assertFalse(isPointInsidePolygon((-0.001, 0.5), square))
			self.assertTrue(isPointInsidePolygon((0.5, 0.001), square))
			self.assertFalse(isPointInsidePolygon((0.5, -0.001), square))

			# This test asserts that a square has the correct number of vertices.
			self.assertEqual(len(square.vertices), square.N)

			# This test asserts that isConvex() works for any closed arrangement of vertices.
			self.assertTrue(square.convex)

			# This test asserts that convexNormalisation produces the correct output.
			self.assertFalse(False in np.equal(
				convexNormalisation(square),
				np.array([[0., 0.5], [0.5, 1.], [1., 0.5], [0.5, 0.]]),
			))
		# This test asserts that after convexNormalisation, the two squares produce the same output.
		self.assertFalse(False in np.equal(convexNormalisation(squares[0]), convexNormalisation(squares[1])))
		# This test asserts that after convexNormalisation, the quads produce the same output.
		for quad in quads:
			quad.vertices = convexNormalisation(quad)
		self.assertFalse(False in np.equal(quads[0].vertices, quads[1].vertices))
		# np.allclose is used, as opposed to np.equal, to account for floating point errors.
		self.assertTrue(np.allclose(quads[0].vertices, quads[2].vertices))
		self.assertTrue(np.allclose(quads[0].vertices, quads[3].vertices))

	def test_random_polygon(self) -> None:
		'''
		Stress test multiple properties of the class RandomPolygon.
		'''

		# This test asserts that generateConvexPolygon always produces a unique output.
		for i in range(100):
			self.assertFalse(np.all(np.equal(generateConvexPolygon(3), generateConvexPolygon(3))))

		for i in range(10000):
			polygon = RandomPolygon(20, allow_concave=True)
			LV = largestVector(polygon)

			# This test asserts that a polygon has the correct number of vertices.
			self.assertEqual(len(polygon.vertices), polygon.N)

			# This test asserts that the vertices are strictly bounded between 0.0 and 1.0.
			self.assertEqual(polygon.vertices.min(), 0.)
			self.assertEqual(polygon.vertices.max(), 1.)

			# This test asserts that the largest vector is of magnitude 1.0.
			self.assertEqual(LV[0], 1.)

			# This test asserts that the area(), used for calculating the area of a polygon is accurate to at least 6 decimal
			# places. This comparison is bounded due to the area() being 64-bit, whilst the comparison function,
			# cv2.contourArea(), is 32-bit.
			self.assertAlmostEqual(
				polygon.area,
				cv2.contourArea(polygon.vertices.astype('float32')),
				places=7,
			)

			# This test asserts that no 3 adjacent vertices are colinear.
			for n in range(polygon.N):
				self.assertFalse(isColinear(np.array([
					polygon.vertices[n - 1 if n > 0 else polygon.N - 1],
					polygon.vertices[n],
					polygon.vertices[(n + 1) % polygon.N],
				])))

			if polygon.convex:
				# This test asserts that all supposedly convex polygons are in fact convex. As a result, if this test passes, we
				# can assume that the generateConvexPolygon() function works as intended.
				self.assertTrue(polygon.convex)

				# This test asserts that the largest vector lies across the x-axis.
				self.assertTrue(polygon.vertices[LV[1][0]][0] == 0.)
				self.assertTrue(polygon.vertices[LV[1][1]][0] == 1.)

				# This test asserts that the calculated centroid lies within the polygon. For concave shapes, this test may fail.
				isPointInsidePolygon(polygon.centroid, polygon)
				self.assertEqual(drawPolygon(polygon, 100)[
					round(polygon.centroid[0] * 99),
					round(polygon.centroid[1] * 99),
				], 1)

				# This test asserts that convexNormalisation does not continuously alter the polygon.
				# np.allclose is used, as opposed to np.equal, to account for floating point errors.
				self.assertTrue(np.allclose(polygon.vertices, convexNormalisation(polygon)))

	def test_unit_polygon(self) -> None:
		'''
		Test used in conjunction with ./unit_polygons.py.
		'''

		# Test the vertices, centroid and area of the UnitRectangle for varying epsilons.
		for [epsilon, vertices] in [
			(1., [[0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5]]),
			(0.5, [[0.25, 1.], [0.25, -1.], [-0.25, -1.], [-0.25, 1.]]),
			(1.25, [[0.625, 0.4], [0.625, -0.4], [-0.625, -0.4], [-0.625, 0.4]]),
		]:
			R = UnitRectangle(epsilon)
			P = Polygon(vertices)
			self.assertTrue(np.all(np.equal(R.vertices, P.vertices)))
			self.assertEqual(R.area, P.area)
			self.assertEqual(centroid(R), (0., 0.))

		# Test the vertices and area of the UnitTriangle for varying r, theta.
		for [r, theta] in [
			(1., np.pi / 2.), # equilateral
			(0.5, np.pi / 2.),
			(1., 1.),
			(1., 2.),
			(1., 3.),
			(1., 4.),
			(1., 5.),
			(1., 6.),
		]:
			T = UnitTriangle(1., np.pi / 3)
			P = Polygon(T.vertices)
			self.assertAlmostEqual(T.area, P.area)
			self.assertEqual(T.vertices[:, 0].min() + T.vertices[:, 0].max(), 0.)
			self.assertEqual(T.vertices[:, 1].min() + T.vertices[:, 1].max(), 0.)
