/*
Generate python bindings for functions in `/kac_core/geometry` and configure C++ type conversions.
*/

// core
#include <array>
#include <random>
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
	for (int i = 0; i < P.size(); i++) { out.push_back({{P[i].x, P[i].y}}); }
	return out;
}

T::Polygon convertVectorToPolygon(const _Vertices& V) {
	T::Polygon out(V.size());
	for (int i = 0; i < V.size(); i++) { out[i] = T::Point(V[i][0], V[i][1]); }
	return out;
}

/*
PyBind11 exports.
*/

_Point _centroid(const _Vertices& V, double area) {
	T::Point c = g::centroid(convertVectorToPolygon(V), area);
	return {c.x, c.y};
}

_Vertices _normaliseConvexPolygon(const _Vertices& V) {
	return convertPolygonToVector(g::normaliseConvexPolygon(convertVectorToPolygon(V)));
}

_Vertices _generateConvexPolygon(const int& N) {
	return convertPolygonToVector(g::generateConvexPolygon(N));
}

bool _isColinear(const std::array<_Point, 3>& V) {
	return g::isColinear(
		T::Point(V[0][0], V[0][1]), T::Point(V[1][0], V[1][1]), T::Point(V[2][0], V[2][1])
	);
}

bool _isConvex(const _Vertices& V) { return g::isConvex(convertVectorToPolygon(V)); }

bool _isPointInsideConvexPolygon(const _Point& p, const _Vertices& V) {
	return g::isPointInsideConvexPolygon(T::Point(p[0], p[1]), convertVectorToPolygon(V));
}

std::pair<double, std::pair<int, int>> _largestVector(const _Vertices& V) {
	return g::largestVector(convertVectorToPolygon(V));
}

std::pair<bool, _Point> _lineIntersection(_Line& A, _Line& B) {
	std::pair<bool, T::Point> out = g::lineIntersection(
		T::Line(T::Point(A[0][0], A[0][1]), T::Point(A[1][0], A[1][1])),
		T::Line(T::Point(B[0][0], B[0][1]), T::Point(B[1][0], B[1][1]))
	);
	return std::make_pair(out.first, _Point({out.second.x, out.second.y}));
}

double _polygonArea(const _Vertices& V) { return g::polygonArea(convertVectorToPolygon(V)); }

/*
PyBind11 config.
*/

PYBIND11_MODULE(_geometry, m) {
	m.doc() = "_geometry";
	m.def("_centroid", &_centroid);
	m.def("_generateConvexPolygon", &_generateConvexPolygon);
	m.def("_isColinear", &_isColinear);
	m.def("_isConvex", &_isConvex);
	m.def("_isPointInsideConvexPolygon", &_isPointInsideConvexPolygon);
	m.def("_largestVector", &_largestVector);
	m.def("_lineIntersection", &_lineIntersection);
	m.def("_normaliseConvexPolygon", &_normaliseConvexPolygon);
	m.def("_polygonArea", &_polygonArea);
}