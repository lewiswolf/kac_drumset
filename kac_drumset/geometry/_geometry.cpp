/*
Generate python bindings for functions in `/lib` and configure C++ type
conversions.
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

std::vector<std::array<double, 2>> convertVerticesToVector(const T::Vertices& V
) {
	std::vector<std::array<double, 2>> out;
	for (int i = 0; i < V.size(); i++) { out.push_back({{V[i].x, V[i].y}}); }
	return out;
}

T::Vertices convertVectorToVertices(const std::vector<std::array<double, 2>>& V
) {
	T::Vertices out(V.size());
	for (int i = 0; i < V.size(); i++) { out[i] = T::Point(V[i][0], V[i][1]); }
	return out;
}

/*
PyBind11 exports.
*/

std::pair<double, double>
_centroid(const std::vector<std::array<double, 2>>& V, double area) {
	T::Point c = g::centroid(convertVectorToVertices(V), area);
	return std::make_pair(c.x, c.y);
}

std::vector<std::array<double, 2>>
_convexNormalisation(const std::vector<std::array<double, 2>>& V) {
	return convertVerticesToVector(
		g::convexNormalisation(convertVectorToVertices(V))
	);
}

std::vector<std::array<double, 2>> _generateConvexPolygon(const int& N) {
	return convertVerticesToVector(g::generateConvexPolygon(N));
}

bool _isColinear(const std::array<std::array<double, 2>, 3>& V) {
	return g::isColinear(
		T::Point(V[0][0], V[0][1]),
		T::Point(V[1][0], V[1][1]),
		T::Point(V[2][0], V[2][1])
	);
}

bool _isConvex(const std::vector<std::array<double, 2>>& V) {
	return g::isConvex(convertVectorToVertices(V));
}

std::pair<double, std::pair<int, int>>
_largestVector(const std::vector<std::array<double, 2>>& V) {
	return g::largestVector(convertVectorToVertices(V));
}

double _polygonArea(const std::vector<std::array<double, 2>>& V) {
	return g::polygonArea(convertVectorToVertices(V));
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
	m.def("_largestVector", &_largestVector);
	m.def("_polygonArea", &_polygonArea);
}