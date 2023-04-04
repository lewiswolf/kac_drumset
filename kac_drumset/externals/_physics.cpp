/*
Generate python bindings for functions in `/kac_core/physics`.
*/

// dependencies
#include <kac_core.hpp>
#include <pybind11/pybind11.h>	  // python bindings
#include <pybind11/stl.h>		  // type conversion

namespace py = pybind11;
namespace p = kac_core::physics;
namespace T = kac_core::types;

T::Matrix_1D _FDTDWaveform2D(
	T::Matrix_2D u_0,
	T::Matrix_2D u_1,
	const T::BooleanImage& B,
	const double& c_0,
	const double& c_1,
	const double& c_2,
	const unsigned long& T,
	const std::array<double, 2>& w
) {
	return p::FDTDWaveform2D(u_0, u_1, B, c_0, c_1, c_2, T, T::Point(w[0], w[1]));
}

PYBIND11_MODULE(_physics, m) {
	m.doc() = "_physics";
	m.def("_circularAmplitudes", &p::circularAmplitudes);
	m.def("_circularSeries", &p::circularSeries);
	m.def("_equilateralTriangleAmplitudes", &p::equilateralTriangleAmplitudes);
	m.def("_equilateralTriangleSeries", &p::equilateralTriangleSeries);
	m.def("_FDTDUpdate2D", &p::FDTDUpdate2D);
	m.def("_FDTDWaveform2D", &_FDTDWaveform2D);
	m.def("_raisedCosine1D", &p::raisedCosine1D);
	m.def("_raisedCosine2D", &p::raisedCosine2D);
	m.def("_raisedTriangle1D", &p::raisedTriangle1D);
	m.def("_raisedTriangle2D", &p::raisedTriangle2D);
	m.def("_rectangularAmplitudes", &p::rectangularAmplitudes);
	m.def("_rectangularSeries", &p::rectangularSeries);
	m.def("_WaveEquationWaveform2D", &p::WaveEquationWaveform2D);
	m.def("besselJ", &p::besselJ);
	m.def("besselJZero", &p::besselJZero);
}
