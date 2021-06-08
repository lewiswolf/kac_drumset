#include <pybind11/pybind11.h>	// python bindings
// #include <GLFW/glfw3.h>		// GPU helper

namespace py = pybind11;

float add_floats(float x, float y) {
	return x + y;
}

PYBIND11_MODULE(physical_lib, handle) {
	handle.doc() = "Physical modelling library";
	handle.def("add", &add_floats);
}