/*
Generate python bindings for functions in `/lib` and configure C++ type
conversions.
*/

// core
#include <array>
#include <vector>

// dependencies
#include "geometry.hpp"			  // my geometry cpp library
#include <pybind11/pybind11.h>	  // python bindings
#include <pybind11/stl.h>		  // type conversion

namespace py = pybind11;
namespace g = geometry;

/*
Type conversions.
*/

std::vector<std::array<double, 2>> convertVerticesToVector(const g::Vertices& V
) {
	// covert to vertices to vector
	std::vector<std::array<double, 2>> out;
	for (int i = 0; i < V.size(); i++) { out.push_back({{V[i].x, V[i].y}}); }
	return out;
}

g::Vertices convertVectorToVertices(const std::vector<std::array<double, 2>>& V
) {
	// covert to vector to vertices
	g::Vertices out(V.size());
	for (int i = 0; i < V.size(); i++) { out[i] = g::Point(V[i][0], V[i][1]); }
	return out;
}

/*
PyBind11 exports.
*/

std::vector<std::array<double, 2>> _generateConvexPolygon(const int& n) {
	return convertVerticesToVector(g::generateConvexPolygon(n));
}

bool _isConvex(const std::vector<std::array<double, 2>>& v) {
	return g::isConvex(convertVectorToVertices(v));
}

PYBIND11_MODULE(_geometry, m) {
	m.doc() = "_geometry";
	m.def("_generateConvexPolygon", &_generateConvexPolygon);
	m.def("_isConvex", &_isConvex);
}