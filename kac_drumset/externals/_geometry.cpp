/*
Generate python bindings for functions in `/kac_core/geometry` and configure C++ type conversions.
*/

// core
#include <array>
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
Type conversions.
*/

std::vector<std::array<double, 2>> convertPolygonToVector(const T::Polygon& P) {
	std::vector<std::array<double, 2>> out;
	for (int i = 0; i < P.size(); i++) { out.push_back({{P[i].x, P[i].y}}); }
	return out;
}

T::Polygon convertVectorToPolygon(const std::vector<std::array<double, 2>>& V) {
	T::Polygon out(V.size());
	for (int i = 0; i < V.size(); i++) { out[i] = T::Point(V[i][0], V[i][1]); }
	return out;
}

/*
PyBind11 exports.
*/

std::array<double, 2> _centroid(const std::vector<std::array<double, 2>>& V, double area) {
	T::Point c = g::centroid(convertVectorToPolygon(V), area);
	return {c.x, c.y};
}

std::vector<std::array<double, 2>> _convexNormalisation(const std::vector<std::array<double, 2>>& V
) {
	return convertPolygonToVector(g::convexNormalisation(convertVectorToPolygon(V)));
}

std::vector<std::array<double, 2>> _generateConvexPolygon(const int& N) {
	return convertPolygonToVector(g::generateConvexPolygon(N));
}

bool _isColinear(const std::array<std::array<double, 2>, 3>& V) {
	return g::isColinear(
		T::Point(V[0][0], V[0][1]), T::Point(V[1][0], V[1][1]), T::Point(V[2][0], V[2][1])
	);
}

bool _isConvex(const std::vector<std::array<double, 2>>& V) {
	return g::isConvex(convertVectorToPolygon(V));
}

bool _isPointInsidePolygon(
	const std::array<double, 2>& p, const std::vector<std::array<double, 2>>& V
) {
	return g::isPointInsidePolygon(T::Point(p[0], p[1]), convertVectorToPolygon(V));
}

std::pair<double, std::pair<int, int>> _largestVector(const std::vector<std::array<double, 2>>& V) {
	return g::largestVector(convertVectorToPolygon(V));
}

double _polygonArea(const std::vector<std::array<double, 2>>& V) {
	return g::polygonArea(convertVectorToPolygon(V));
}

/*
PyBind11 config.
*/

PYBIND11_MODULE(_geometry, m) {
	m.doc() = "_geometry";
	m.def("_centroid", &_centroid);
	m.def("_convexNormalisation", &_convexNormalisation);
	m.def("_generateConvexPolygon", &_generateConvexPolygon);
	m.def("_isColinear", &_isColinear);
	m.def("_isConvex", &_isConvex);
	m.def("_isPointInsidePolygon", &_isPointInsidePolygon);
	m.def("_largestVector", &_largestVector);
	m.def("_polygonArea", &_polygonArea);
}