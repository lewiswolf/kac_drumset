// core
#include <array>

// dependencies
#include "geometry.hpp"
#include <pybind11/pybind11.h>	  // python bindings
#include <pybind11/stl.h>		  // type conversion

namespace py = pybind11;
namespace g = geometry;

std::vector<std::array<double, 2>> convertVerticesToVector(const g::Vertices& V
) {
	/*
	Covert to vector of arrays.
	*/

	std::vector<std::array<double, 2>> out;
	for (int i = 0; i < V.size(); i++) { out.push_back({{V[i].x, V[i].y}}); }
	return out;
}

std::vector<std::array<double, 2>> generateConvexPolygon(const int& n) {
	return convertVerticesToVector(g::generateConvexPolygon(n));
}

PYBIND11_MODULE(_geometry, m) {
	m.doc() = "_geometry";
	m.def("generateConvexPolygon", &generateConvexPolygon);
}