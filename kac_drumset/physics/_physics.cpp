// dependencies
#include <pybind11/pybind11.h>	  // python bindings
#include <pybind11/stl.h>		  // type conversion

// src
#include "raised_cosine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_physics, m) {
	m.doc() = "_physics";
	m.def("_raisedCosine1D", &raisedCosine1D);
	m.def("_raisedCosine2D", &raisedCosine2D);
}