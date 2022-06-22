/*
Generate python bindings for functions in `/lib` and configure C++ type
conversions.
*/

// core
#include <array>
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

PYBIND11_MODULE(_geometry, m) {
	m.doc() = "_geometry";
	m.def("_generateConvexPolygon", &_generateConvexPolygon);
	m.def("_isColinear", &_isColinear);
	m.def("_isConvex", &_isConvex);
}