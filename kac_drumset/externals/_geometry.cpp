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
	for (std::size_t n = 0; n < P.size(); n++) { out.push_back({{P[n].x, P[n].y}}); }
	return out;
}

T::Polygon convertVectorToPolygon(const _Vertices& V) {
	T::Polygon out;
	for (std::size_t n = 0; n < V.size(); n++) { out.push_back(T::Point(V[n])); }
	return out;
}

/*
PyBind11 config.
*/

PYBIND11_MODULE(_geometry, m) {
	m.doc() = "_geometry";
	m.def("_generateIrregularStar", [](const int& N) -> _Vertices {
		return convertPolygonToVector(g::generateIrregularStar(N));
	});
	m.def("_generateConvexPolygon", [](const int& N) -> _Vertices {
		return convertPolygonToVector(g::generateConvexPolygon(N));
	});
	m.def("_generatePolygon", [](const int& N) -> _Vertices {
		return convertPolygonToVector(g::generatePolygon(N));
	});
	m.def("_generateUnitRectangle", [](const double& epsilon) -> _Vertices {
		return convertPolygonToVector(g::generateUnitRectangle(epsilon));
	});
	// m.def("_generateUnitTriangle", [](const double& r, const double& theta) -> _Vertices {
	// 	return convertPolygonToVector(g::generateUnitTriangle(r, theta));
	// });
	m.def("_isColinear", [](const std::array<_Point, 3>& V) -> bool {
		return g::isColinear(T::Point(V[0]), T::Point(V[1]), T::Point(V[2]));
	});
	m.def("_isConvex", [](const _Vertices& V) -> bool {
		return g::isConvex(convertVectorToPolygon(V));
	});
	m.def("_isPointInsideConvexPolygon", [](const _Point& p, const _Vertices& V) -> bool {
		return g::isPointInsideConvexPolygon(T::Point(p), convertVectorToPolygon(V));
	});
	m.def("_isPointInsidePolygon", [](const _Point& p, const _Vertices& V) -> bool {
		return g::isPointInsidePolygon(T::Point(p), convertVectorToPolygon(V));
	});
	m.def("_isSimple", [](const _Vertices& V) -> bool {
		return g::isSimple(convertVectorToPolygon(V));
	});
	m.def("_largestVector", [](const _Vertices& V) -> std::pair<double, std::pair<int, int>> {
		return g::largestVector(convertVectorToPolygon(V));
	});
	m.def("_lineIntersection", [](_Line& A, _Line& B) -> std::pair<std::string, _Point> {
		std::pair<std::string, T::Point> out = g::lineIntersection(
			T::Line(T::Point(A[0]), T::Point(A[1])), T::Line(T::Point(B[0]), T::Point(B[1]))
		);
		return std::make_pair(out.first, _Point({out.second.x, out.second.y}));
	});
	m.def("_normaliseConvexPolygon", [](const _Vertices& V, const bool& signed_norm) -> _Vertices {
		return convertPolygonToVector(
			g::normaliseConvexPolygon(convertVectorToPolygon(V), signed_norm)
		);
	});
	m.def("_normalisePolygon", [](const _Vertices& V, const bool& signed_norm) -> _Vertices {
		return convertPolygonToVector(g::normalisePolygon(convertVectorToPolygon(V), signed_norm));
	});
	m.def("_normaliseSimplePolygon", [](const _Vertices& V, const bool& signed_norm) -> _Vertices {
		return convertPolygonToVector(
			g::normaliseSimplePolygon(convertVectorToPolygon(V), signed_norm)
		);
	});
	m.def("_polygonArea", [](const _Vertices& V) -> double {
		return g::polygonArea(convertVectorToPolygon(V));
	});
	m.def("_polygonCentroid", [](const _Vertices& V) -> _Point {
		T::Point p = g::polygonCentroid(convertVectorToPolygon(V));
		return {p.x, p.y};
	});
	m.def("_scalePolygonByArea", [](const _Vertices& V, const double& a) -> _Vertices {
		return convertPolygonToVector(g::scalePolygonByArea(convertVectorToPolygon(V), a));
	});
}
