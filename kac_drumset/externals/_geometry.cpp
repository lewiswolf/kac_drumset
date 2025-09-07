/*
Generate python bindings for functions in `/kac_core/geometry` and configure C++ type conversions.
*/

// core
#include <array>
#include <random>
#include <string>
#include <time.h>
#include <utility>
#include <vector>

// dependencies
#include <kac_core.hpp>
#include <pybind11/pybind11.h>	  // python bindings
#include <pybind11/stl.h>		  // type conversion

namespace py = pybind11;
namespace g = kac_core::geometry;
namespace T = kac_core::types;

/*
Intermediary types.
*/

typedef std::array<double, 2> _Point;
typedef std::array<_Point, 2> _Line;
typedef std::vector<_Point> _Vertices;

/*
Type conversions.
*/

_Vertices convertPolygonToVector(const T::Polygon& P) {
	_Vertices out;
	for (unsigned long n = 0; n < P.size(); n++) { out.push_back({{P[n].x, P[n].y}}); }
	return out;
}

T::Polygon convertVectorToPolygon(const _Vertices& V) {
	T::Polygon out;
	for (unsigned long n = 0; n < V.size(); n++) { out.push_back(T::Point(V[n][0], V[n][1])); }
	return out;
}

/*
PyBind11 exports.
*/

_Vertices _generateIrregularStar(const int& N) {
	return convertPolygonToVector(g::generateIrregularStar(N));
}

_Vertices _generateConvexPolygon(const int& N) {
	return convertPolygonToVector(g::generateConvexPolygon(N));
}

_Vertices _generatePolygon(const int& N) { return convertPolygonToVector(g::generatePolygon(N)); }

_Vertices _generateUnitRectangle(const double& epsilon) {
	return convertPolygonToVector(g::generateUnitRectangle(epsilon));
}

// _Vertices _generateUnitTriangle(const double& r, const double& theta) {
// 	return convertPolygonToVector(g::generateUnitTriangle(r, theta));
// }

bool _isColinear(const std::array<_Point, 3>& V) {
	return g::isColinear(
		T::Point(V[0][0], V[0][1]), T::Point(V[1][0], V[1][1]), T::Point(V[2][0], V[2][1])
	);
}

bool _isConvex(const _Vertices& V) { return g::isConvex(convertVectorToPolygon(V)); }

bool _isPointInsideConvexPolygon(const _Point& p, const _Vertices& V) {
	return g::isPointInsideConvexPolygon(T::Point(p[0], p[1]), convertVectorToPolygon(V));
}

bool _isPointInsidePolygon(const _Point& p, const _Vertices& V) {
	return g::isPointInsidePolygon(T::Point(p[0], p[1]), convertVectorToPolygon(V));
}

bool _isSimple(const _Vertices& V) { return g::isSimple(convertVectorToPolygon(V)); }

std::pair<double, std::pair<int, int>> _largestVector(const _Vertices& V) {
	return g::largestVector(convertVectorToPolygon(V));
}

std::pair<std::string, _Point> _lineIntersection(_Line& A, _Line& B) {
	std::pair<std::string, T::Point> out = g::lineIntersection(
		T::Line(T::Point(A[0][0], A[0][1]), T::Point(A[1][0], A[1][1])),
		T::Line(T::Point(B[0][0], B[0][1]), T::Point(B[1][0], B[1][1]))
	);
	return std::make_pair(out.first, _Point({out.second.x, out.second.y}));
}

_Vertices _normaliseConvexPolygon(const _Vertices& V, const bool& signed_norm) {
	return convertPolygonToVector(
		g::normaliseConvexPolygon(convertVectorToPolygon(V), signed_norm)
	);
}

_Vertices _normalisePolygon(const _Vertices& V, const bool& signed_norm) {
	return convertPolygonToVector(g::normalisePolygon(convertVectorToPolygon(V), signed_norm));
}

_Vertices _normaliseSimplePolygon(const _Vertices& V, const bool& signed_norm) {
	return convertPolygonToVector(
		g::normaliseSimplePolygon(convertVectorToPolygon(V), signed_norm)
	);
}

double _polygonArea(const _Vertices& V) { return g::polygonArea(convertVectorToPolygon(V)); }

_Point _polygonCentroid(const _Vertices& V) {
	T::Point p = g::polygonCentroid(convertVectorToPolygon(V));
	return {p.x, p.y};
}

_Vertices _scalePolygonByArea(const _Vertices& V, const double& a) {
	return convertPolygonToVector(g::scalePolygonByArea(convertVectorToPolygon(V), a));
}

/*
PyBind11 config.
*/

PYBIND11_MODULE(_geometry, m) {
	m.doc() = "_geometry";
	m.def("_generateIrregularStar", &_generateIrregularStar);
	m.def("_generatePolygon", &_generatePolygon);
	m.def("_generateConvexPolygon", &_generateConvexPolygon);
	m.def("_generateUnitRectangle", &_generateUnitRectangle);
	// m.def("_generateUnitTriangle", &_generateUnitTriangle);
	m.def("_isColinear", &_isColinear);
	m.def("_isConvex", &_isConvex);
	m.def("_isPointInsideConvexPolygon", &_isPointInsideConvexPolygon);
	m.def("_isPointInsidePolygon", &_isPointInsidePolygon);
	m.def("_isSimple", &_isSimple);
	m.def("_largestVector", &_largestVector);
	m.def("_lineIntersection", &_lineIntersection);
	m.def("_normaliseConvexPolygon", &_normaliseConvexPolygon);
	m.def("_normalisePolygon", &_normalisePolygon);
	m.def("_normaliseSimplePolygon", &_normaliseSimplePolygon);
	m.def("_polygonArea", &_polygonArea);
	m.def("_polygonCentroid", &_polygonCentroid);
	m.def("_scalePolygonByArea", &_scalePolygonByArea);
}
