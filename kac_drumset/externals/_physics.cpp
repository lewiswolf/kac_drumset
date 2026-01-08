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

PYBIND11_MODULE(_physics, m) {
	m.doc() = "_physics";
	m.def("_AdditiveSynthesis1D", &p::AdditiveSynthesis1D);
	m.def("_AdditiveSynthesis2D", &p::AdditiveSynthesis2D);
	m.def("_circularAmplitudes", &p::circularAmplitudes);
	m.def("_circularChladniPattern", &p::circularChladniPattern);
	m.def("_circularSeries", &p::circularSeries);
	m.def("_equilateralTriangleAmplitudes", &p::equilateralTriangleAmplitudes);
	m.def("_equilateralTriangleSeries", &p::equilateralTriangleSeries);
	m.def("_FDTDUpdate2D", &p::FDTDUpdate2D);
	m.def(
		"_FDTDWaveform2D",
		[](T::Matrix_2D u_0,
		   T::Matrix_2D u_1,
		   const T::BooleanImage& B,
		   const double& c_0,
		   const double& c_1,
		   const double& c_2,
		   const unsigned long& T,
		   const std::array<double, 2>& w) -> T::Matrix_1D {
			return p::FDTDWaveform2D(u_0, u_1, B, c_0, c_1, c_2, T, T::Point(w));
		}
	);
	m.def("_raisedCosine1D", &p::raisedCosine1D);
	m.def(
		"_raisedCosine2D",
		[](const std::array<double, 2>& mu,
		   const double& sigma,
		   const std::size_t& size_X,
		   const std::size_t& size_Y) -> T::Matrix_2D {
			return p::raisedCosine2D(T::Point(mu), sigma, size_X, size_Y);
		}
	);
	m.def("_raisedTriangle1D", &p::raisedTriangle1D);
	m.def(
		"_raisedTriangle2D",
		[](const std::array<double, 2>& mu,
		   const double& x_a,
		   const double& x_b,
		   const double& y_a,
		   const double& y_b,
		   const std::size_t& size_X,
		   const std::size_t& size_Y) -> T::Matrix_2D {
			return p::raisedTriangle2D(T::Point(mu), x_a, x_b, y_a, y_b, size_X, size_Y);
		}
	);
	m.def("_linearAmplitudes", &p::linearAmplitudes);
	m.def("_linearSeries", &p::linearSeries);
	m.def("_rectangularAmplitudes", &p::rectangularAmplitudes);
	m.def("_rectangularChladniPattern", &p::rectangularChladniPattern);
	m.def("_rectangularSeries", &p::rectangularSeries);
	m.def("besselJ", &p::besselJ);
	m.def("besselJZero", &p::besselJZero);
}
