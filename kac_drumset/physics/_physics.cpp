// dependencies
#include <kac_core.hpp>
#include <pybind11/pybind11.h>	  // python bindings
#include <pybind11/stl.h>		  // type conversion

namespace py = pybind11;
namespace p = kac_core::physics;

PYBIND11_MODULE(_physics, m) {
	m.doc() = "_physics";
	m.def("_FDTDWaveform2D", &p::FDTDWaveform2D);
	m.def("_raisedCosine1D", &p::raisedCosine1D);
	m.def("_raisedCosine2D", &p::raisedCosine2D);
	m.def("besselJ", &p::besselJ);
	m.def("besselJZero", &p::besselJZero);
}