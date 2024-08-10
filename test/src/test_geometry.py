# core
import math
import random
from unittest import TestCase

# dependencies
import cv2					# image processing
import numpy as np 			# maths

# src
from kac_drumset.externals._geometry import (
	_generateConvexPolygon,
	_generateIrregularStar,
	_generatePolygon,
	_isConvex,
	_isPointInsideConvexPolygon,
	# _isPointInsidePolygon,
	_normaliseConvexPolygon,
)
from kac_drumset.geometry import (
	# methods
	isColinear,
	largestVector,
	lineIntersection,
	# classes
	Circle,
	ConvexPolygon,
	IrregularStar,
	TravellingSalesmanPolygon,
	UnitRectangle,
	# UnitTriangle,
	# types
	Ellipse,
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

		C = Circle(1.)

		# This test asserts that the area of a unit circle is calculated correctly.
		self.assertEqual(C.area, np.pi)

		# This test asserts that the area is correct for both the interior and boundaries.
		for _ in range(50):
			r_1 = random.uniform(0., 1.)
			r_2 = random.uniform(0., 1.) + 1.
			theta = random.uniform(0., 2 * np.pi)
			self.assertTrue(C.isPointInside((r_1 * math.cos(theta), r_1 * math.sin(theta))))
			self.assertFalse(C.isPointInside((r_2 * math.cos(theta), r_2 * math.sin(theta))))

		for r in [0.1, 0.25, 0.5, 1., 2.]:
			C = Circle(r)

			# This test asserts that the circle correctly has equal foci.
			self.assertEqual(C.major, C.minor)
			self.assertEqual(C.r, C.major)
			self.assertEqual(C.r, C.minor)

			# This test asserts that the default centroid is (0., 0.).
			self.assertEqual(C.centroid, (0., 0.))

			# This test asserts that the centroid is within the shape.
			self.assertTrue(C.isPointInside(C.centroid))

			# This test asserts that the default eccentricity is 0.
			self.assertEqual(C.eccentricity(), 0.0)

			# This test asserts that the default focal distance is 0.
			self.assertEqual(C.focalDistance(), 0.0)

			# This test asserts that the default foci are each (0., 0.).
			self.assertEqual(C.foci(), ((0., 0.), (0., 0.)))

			# This test asserts that areas can be properly updated on the fly.
			random_area = random.uniform(0., 100.)
			self.assertNotEqual(C.area, random_area)
			C.area = random_area
			self.assertAlmostEqual(C.area, random_area)

			# This test asserts that the center of the boolean mask is always true.
			M = C.draw(101)
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
			Polygon([[0., 0.], [1.1, 0.], [1., 1.], [0., 1.]]),
			Polygon([[0., 0.], [0., 1.], [1., 1.], [1.1, 0.]]),
			Polygon([[0., 0.], [0., 1.1], [1., 1.], [1., 0.]]),
			Polygon([[0., 0.], [1., 0.], [1., 1.], [0., 1.1]]),
		]

		for P in quads + squares:
			# This test asserts that a square has the correct number of vertices.
			self.assertEqual(len(P.vertices), P.N)

			# This test asserts that _isConvex works for any closed arrangement of vertices.
			self.assertTrue(P.convex)

			# This test asserts that P.isPointInside is True for all vertices.
			for p in P.vertices:
				self.assertTrue(_isPointInsideConvexPolygon(p, P.vertices))
				# self.assertTrue(_isPointInsidePolygon(p, P.vertices))

		for square in squares:
			# This test asserts that P.isPointInside correctly identifies points inside each square.
			# self.assertTrue(_isPointInsidePolygon((0.999, 0.5), square.vertices))
			self.assertTrue(_isPointInsideConvexPolygon((0.999, 0.5), square.vertices))
			# self.assertFalse(_isPointInsidePolygon((1.001, 0.5), square.vertices))
			self.assertFalse(_isPointInsideConvexPolygon((1.001, 0.5), square.vertices))
			# self.assertTrue(_isPointInsidePolygon((0.5, 0.999), square.vertices))
			self.assertTrue(_isPointInsideConvexPolygon((0.5, 0.999), square.vertices))
			# self.assertFalse(_isPointInsidePolygon((0.5, 1.001), square.vertices))
			self.assertFalse(_isPointInsideConvexPolygon((0.5, 1.001), square.vertices))
			# self.assertTrue(_isPointInsidePolygon((0.001, 0.5), square.vertices))
			self.assertTrue(_isPointInsideConvexPolygon((0.001, 0.5), square.vertices))
			# self.assertFalse(_isPointInsidePolygon((-0.001, 0.5), square.vertices))
			self.assertFalse(_isPointInsideConvexPolygon((-0.001, 0.5), square.vertices))
			# self.assertTrue(_isPointInsidePolygon((0.5, 0.001), square.vertices))
			self.assertTrue(_isPointInsideConvexPolygon((0.5, 0.001), square.vertices))
			# self.assertFalse(_isPointInsidePolygon((0.5, -0.001), square.vertices))
			self.assertFalse(_isPointInsideConvexPolygon((0.5, -0.001), square.vertices))

			# These test asserts that the midpoint of each sides are inside the polygon.
			# self.assertTrue(_isPointInsidePolygon((0., 0.5), square.vertices))
			self.assertTrue(_isPointInsideConvexPolygon((0., 0.5), square.vertices))
			# self.assertTrue(_isPointInsidePolygon((1., 0.5), square.vertices))
			self.assertTrue(_isPointInsideConvexPolygon((1., 0.5), square.vertices))
			# self.assertTrue(_isPointInsidePolygon((0.5, 0.), square.vertices))
			self.assertTrue(_isPointInsideConvexPolygon((0.5, 0.), square.vertices))
			# self.assertTrue(_isPointInsidePolygon((0.5, 1.), square.vertices))
			self.assertTrue(_isPointInsideConvexPolygon((0.5, 1.), square.vertices))

			# This test asserts that _normaliseConvexPolygon produces the correct output.
			self.assertFalse(False in np.equal(
				np.array(_normaliseConvexPolygon(square.vertices)),
				np.array([[0., 0.5], [0.5, 1.], [1., 0.5], [0.5, 0.]]),
			))

		# This test asserts that _isSimple works as expected.
		self.assertFalse(Polygon([[0., 0.], [1., 1.], [1., 0.], [0., 1.]]).isSimple())
		for square in squares:
			self.assertTrue(square.isSimple())

		# This test asserts that after _normaliseConvexPolygon, the two squares produce the same output.
		self.assertFalse(False in np.equal(
			_normaliseConvexPolygon(squares[0].vertices),
			_normaliseConvexPolygon(squares[1].vertices)),
		)

		# This test asserts that after _normaliseConvexPolygon, the quads produce the same output.
		for quad in quads:
			quad.vertices = np.array(_normaliseConvexPolygon(quad.vertices))
		self.assertFalse(False in np.equal(quads[0].vertices, quads[1].vertices))
		# np.allclose is used, as opposed to np.equal, to account for floating point errors.
		self.assertTrue(np.allclose(quads[0].vertices, quads[2].vertices))
		self.assertTrue(np.allclose(quads[0].vertices, quads[3].vertices))

	def test_ellipse(self) -> None:
		'''
		Test properties of the type Ellipse.
		'''

		# This test asserts that the major axis is always greater than the minor axis.
		E = Ellipse(0.1, 10.)
		self.assertLessEqual(E.minor, E.major)

		for minor in [0.05, 0.1, 0.25, 0.5, 0.75, 1.]:
			E = Ellipse(minor=minor)

			# This test asserts that the default major is 1.
			self.assertEqual(E.major, 1.)
			self.assertLessEqual(E.minor, 1.)

			# This test asserts that the default centroid is (0., 0.).
			self.assertEqual(E.centroid, (0., 0.))

			# This test asserts that the centroid is within the shape.
			self.assertTrue(E.isPointInside(E.centroid))

			# This test asserts that the default eccentricity is less than 1.
			self.assertLessEqual(E.eccentricity(), 1.)

			# This test asserts that the default focal distance is less than 1.
			self.assertLessEqual(E.focalDistance(), 1.)

			# This test asserts that the default foci lie along the x axis.
			self.assertEqual(E.foci()[0][0], 0.)
			self.assertEqual(E.foci()[1][0], 0.)

			# This test asserts that areas can be properly updated on the fly.
			random_area = random.uniform(0., 100.)
			self.assertNotEqual(E.area, random_area)
			E.area = random_area
			self.assertAlmostEqual(E.area, random_area)

			# This test asserts that the center of the boolean mask is always true.
			M = E.draw(101)
			self.assertEqual(M[50, 50], 1)
			self.assertEqual(M[0, 0], 0)
			self.assertEqual(M[0, 100], 0)
			self.assertEqual(M[100, 0], 0)
			self.assertEqual(M[100, 100], 0)

	def test_lines(self) -> None:
		'''
		Test properties of lines and curves.
		'''

		# This test asserts that lineIntersection() correctly reports none.
		does_it_cross, cross_point = lineIntersection(
			np.array([[0., 0.], [1., 0.]]),
			np.array([[0., 1.], [1., 1.]]),
		)
		self.assertEqual(does_it_cross, 'none')
		self.assertTrue(cross_point[0] == 0. and cross_point[1] == 0.)

		# This test asserts that lineIntersection() correctly reports intersection.
		does_it_cross, cross_point = lineIntersection(
			np.array([[0., 0.], [1., 0.]]),
			np.array([[0.5, -0.5], [0.5, 0.5]]),
		)
		self.assertTrue(cross_point[0] == 0.5 and cross_point[1] == 0.)

		# This test asserts that lineIntersection() correctly reports vertex.
		does_it_cross, cross_point = lineIntersection(
			np.array([[0., 0.], [1., 0.]]),
			np.array([[1., 0.], [1., 1.]]),
		)
		self.assertEqual(does_it_cross, 'vertex')
		self.assertTrue(cross_point[0] == 1. and cross_point[1] == 0.)
		does_it_cross, cross_point = lineIntersection(
			np.array([[0., 0.], [1., 1.]]),
			np.array([[1., 0.], [1., 1.]]),
		)
		self.assertEqual(does_it_cross, 'vertex')
		self.assertTrue(cross_point[0] == 1. and cross_point[1] == 1.)
		does_it_cross, cross_point = lineIntersection(
			np.array([[0., 0.], [1., 0.]]),
			np.array([[0., 0.], [1., 1.]]),
		)
		self.assertEqual(does_it_cross, 'vertex')
		self.assertTrue(cross_point[0] == 0. and cross_point[1] == 0.)
		does_it_cross, cross_point = lineIntersection(
			np.array([[0., 1.], [1., 1.]]),
			np.array([[0., 0.], [0., 1.]]),
		)
		self.assertEqual(does_it_cross, 'vertex')
		self.assertTrue(cross_point[0] == 0. and cross_point[1] == 1.)

		# This test asserts that lineIntersection() correctly reports adjacent.
		does_it_cross, cross_point = lineIntersection(
			np.array([[0., 0.], [1., 0.]]),
			np.array([[0.5, 0.], [0.5, 1.]]),
		)
		self.assertEqual(does_it_cross, 'adjacent')
		self.assertTrue(cross_point[0] == 0.5 and cross_point[1] == 0.)
		does_it_cross, cross_point = lineIntersection(
			np.array([[0.5, 0.], [0.5, 1.]]),
			np.array([[0., 0.], [1., 0.]]),
		)
		self.assertEqual(does_it_cross, 'adjacent')
		self.assertTrue(cross_point[0] == 0.5 and cross_point[1] == 0.)

		# This test asserts that lineIntersection() correctly reports colinear.
		does_it_cross, cross_point = lineIntersection(
			# B inside
			np.array([[0., 0.], [1., 0.]]),
			np.array([[0.25, 0.], [0.75, 0.]]),
		)
		self.assertEqual(does_it_cross, 'colinear')
		self.assertTrue(cross_point[0] == 0.5 and cross_point[1] == 0.)
		does_it_cross, cross_point = lineIntersection(
			# B left
			np.array([[0., 0.], [1., 0.]]),
			np.array([[-0.5, 0.], [0.5, 0.]]),
		)
		self.assertEqual(does_it_cross, 'colinear')
		self.assertTrue(cross_point[0] == 0.25 and cross_point[1] == 0.)
		does_it_cross, cross_point = lineIntersection(
			# B right
			np.array([[0., 0.], [1., 0.]]),
			np.array([[0.5, 0.], [1.5, 0.]]),
		)
		self.assertEqual(does_it_cross, 'colinear')
		self.assertTrue(cross_point[0] == 0.75 and cross_point[1] == 0.)
		does_it_cross, cross_point = lineIntersection(
			# A inside
			np.array([[0.25, 0.], [0.75, 0.]]),
			np.array([[0., 0.], [1., 0.]]),
		)
		self.assertEqual(does_it_cross, 'colinear')
		self.assertTrue(cross_point[0] == 0.5 and cross_point[1] == 0.)
		does_it_cross, cross_point = lineIntersection(
			# A left
			np.array([[-0.5, 0.], [0.5, 0.]]),
			np.array([[0., 0.], [1., 0.]]),
		)
		self.assertEqual(does_it_cross, 'colinear')
		self.assertTrue(cross_point[0] == 0.25 and cross_point[1] == 0.)
		does_it_cross, cross_point = lineIntersection(
			# A right
			np.array([[0.5, 0.], [1.5, 0.]]),
			np.array([[0., 0.], [1., 0.]]),
		)
		self.assertEqual(does_it_cross, 'colinear')
		self.assertTrue(cross_point[0] == 0.75 and cross_point[1] == 0.)
		does_it_cross, cross_point = lineIntersection(
			# Line outside
			np.array([[0., 0.], [1., 0.]]),
			np.array([[1.5, 0.], [2.5, 0.]]),
		)
		self.assertEqual(does_it_cross, 'none')
		self.assertTrue(cross_point[0] == 0. and cross_point[1] == 0.)
		does_it_cross, cross_point = lineIntersection(
			# Line outside
			np.array([[1.5, 0.], [2.5, 0.]]),
			np.array([[0., 0.], [1., 0.]]),
		)
		self.assertEqual(does_it_cross, 'none')
		self.assertTrue(cross_point[0] == 0. and cross_point[1] == 0.)

	def test_random_polygon(self) -> None:
		'''
		Stress test multiple properties of random polygons.
		'''

		for _ in range(10000):
			# This test asserts that all polygon generation methods always produces a unique output.
			self.assertFalse(np.all(np.equal(_generateConvexPolygon(3), _generateConvexPolygon(3))))
			self.assertFalse(np.all(np.equal(_generateIrregularStar(3), _generateIrregularStar(3))))
			self.assertFalse(np.all(np.equal(_generatePolygon(3), _generatePolygon(3))))

			for P in [
				ConvexPolygon,
				IrregularStar,
				TravellingSalesmanPolygon,
			]:
				polygon = P(max_vertices=20)
				LV = largestVector(polygon.vertices)

				# This test asserts that a polygon has the correct number of vertices.
				self.assertEqual(len(polygon.vertices), polygon.N)

				# This test asserts that a polygon is simple.
				self.assertTrue(polygon.isSimple())

				# This test asserts that the vertices are strictly bounded between 0.0 and 1.0.
				self.assertEqual(polygon.vertices.min(), 0.)
				self.assertEqual(polygon.vertices.max(), 1.)

				# This test asserts that the largest vector is of magnitude 1.0.
				self.assertEqual(LV[0], 1.)

				# This test asserts that the area of a polygon is accurate to at least 6 decimal places. This comparison is bounded
				# due to the area being 64-bit, whilst the comparison function, cv2.contourArea(), is 32-bit.
				self.assertAlmostEqual(
					polygon.area,
					cv2.contourArea(polygon.vertices.astype('float32')),
					places=6,
				)

				# This test asserts that no 3 adjacent vertices are colinear.
				for n in range(polygon.N):
					self.assertFalse(isColinear(np.array([
						polygon.vertices[n - 1 if n > 0 else polygon.N - 1],
						polygon.vertices[n],
						polygon.vertices[(n + 1) % polygon.N],
					])))

				# This test asserts that isPointInsidePolygon correctly recognises points outside of the polygon.
				# self.assertFalse(_isPointInsidePolygon((-0.01, -0.01), polygon.vertices))
				# self.assertFalse(_isPointInsidePolygon((2., 2.), polygon.vertices))

				# This test asserts that isPointInsidePolygon includes the vertices.
				# for p in polygon.vertices:
				# 	self.assertTrue(_isPointInsidePolygon(p, polygon.vertices))

				if polygon.convex:
					# This test asserts that all supposedly convex polygons are in fact convex. As a result, if this test passes, we
					# can assume that the _generateConvexPolygon() function works as intended.
					self.assertTrue(_isConvex(polygon.vertices))

					# This test asserts that the largest vector lies across the x-axis.
					self.assertTrue(polygon.vertices[LV[1][0]][0] == 0.)
					self.assertTrue(polygon.vertices[LV[1][1]][0] == 1.)

					# This test asserts that isPointInsideConvexPolygon correctly recognises points outside of the polygon.
					self.assertFalse(_isPointInsideConvexPolygon((-0.01, -0.01), polygon.vertices))
					self.assertFalse(_isPointInsideConvexPolygon((2., 2.), polygon.vertices))

					# This test asserts that isPointInsideConvexPolygon includes the vertices.
					for p in polygon.vertices:
						self.assertTrue(_isPointInsideConvexPolygon(p, polygon.vertices))

					# This test asserts that the calculated centroid lies within the polygon. For concave shapes, this test may fail.
					centroid = polygon.centroid
					self.assertTrue(_isPointInsideConvexPolygon(centroid, polygon.vertices))
					# self.assertTrue(_isPointInsidePolygon(centroid, polygon.vertices))
					self.assertEqual(polygon.draw(100)[
						round(centroid[0] * 99),
						round(centroid[1] * 99),
					], 1)

					# This test asserts that _normaliseConvexPolygon does not continuously alter the polygon.
					# np.allclose is used, as opposed to np.equal, to account for floating point errors.
					self.assertTrue(np.allclose(polygon.vertices, np.array(_normaliseConvexPolygon(polygon.vertices))))

				# This test asserts that polygon translation works as expected.
				polygon.centroid = (10., 10.)
				self.assertAlmostEqual(polygon.centroid[0], 10.)
				self.assertAlmostEqual(polygon.centroid[1], 10.)

	def test_unit_polygon(self) -> None:
		'''
		Test used in conjunction with ./unit_polygons.py.
		'''

		# Test the vertices, area and centroid of the UnitRectangle for varying epsilons.
		for [epsilon, vertices] in [
			(1., [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]]),
			(0.5, [[-0.25, -1.], [-0.25, 1.], [0.25, 1.], [0.25, -1.]]),
			(1.25, [[-0.625, -0.4], [-0.625, 0.4], [0.625, 0.4], [0.625, -0.4]]),
		]:
			R = UnitRectangle(epsilon)
			P = Polygon(vertices)
			self.assertTrue(np.all(np.equal(R.vertices, P.vertices)))
			self.assertEqual(R.area, 1.)
			self.assertEqual(R.centroid, (0., 0.))

		# Test the vertices and area of the UnitTriangle for varying r, theta.
		# for [r, theta] in [
		# 	(0.5, np.pi / 2.),
		# 	(1., 1.),
		# 	(1., 2.),
		# 	(1., 3.),
		# 	(1., 4.),
		# 	(1., 5.),
		# 	(-1., 6.),
		# ]:
		# 	T = UnitTriangle(r, theta)
		# 	P = Polygon(T.vertices)
		# 	self.assertAlmostEqual(T.area, P.area)
		# 	self.assertAlmostEqual(T.area, 1.)

		# 	# This test asserts that the longest line of the UnitTriangle is along the x axis.
		# 	self.assertEqual(T.vertices[:, 0].min() + T.vertices[:, 0].max(), 0.)
		# 	self.assertEqual(T.vertices[:, 1].min() + T.vertices[:, 1].max(), 0.)

		# # This tests asserts the symmetry of the method used to generate UnitTriangle
		# norm_tri = _normaliseConvexPolygon(UnitTriangle(1., 1.).vertices)
		# self.assertTrue(np.all(np.allclose(_normaliseConvexPolygon(UnitTriangle(1., np.pi - 1.).vertices), norm_tri)))
		# self.assertTrue(np.all(np.allclose(_normaliseConvexPolygon(UnitTriangle(1., np.pi + 1.).vertices), norm_tri)))
		# self.assertTrue(np.all(np.allclose(_normaliseConvexPolygon(UnitTriangle(1., -1.).vertices), norm_tri)))

		# # This test asserts that the equilateral triangle is properly constructed.
		# T = UnitTriangle(1., np.pi / 2)
		# for n in range(3):
		# 	a = T.vertices[n]
		# 	b = T.vertices[(n + 1) % 3]
		# 	self.assertEqual(
		# 		((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2) ** 0.5,
		# 		(4 / (3 ** 0.5)) ** 0.5,
		# 	)
