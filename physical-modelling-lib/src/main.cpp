#include <pybind11/pybind11.h>	// python bindings
#include <GL/glew.h>			// OpenGL
#include <GLFW/glfw3.h>

// python namespace
namespace py = pybind11;

float add_floats(float x, float y) {
	return x + y;
}

// configure python bindings
PYBIND11_MODULE(physical_lib, handle) {
	handle.doc() = "Physical modelling library";
	handle.def("add", &add_floats);
}