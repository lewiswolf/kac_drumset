// core
#include <string>
// includes
#include <GL/glew.h>    	// OpenGL
#include <GLFW/glfw3.h>		// OpenGL Window
// src
#include "openGL_functions.h"

int main() {
	std::string windowName = "Drum Model";
	int domainSize[2] = {80, 80}; 	// grid resolution
	float magnifier = 10;			// pixels per grid grid point (1 * 1 => n * n)

	// glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	// glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	// glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	// glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = initOpenGL(domainSize[0], domainSize[1], windowName, magnifier);
	
	// where the shaders are
	// const char* vertex_path_fbo = {"shaders_FDTD/fbo_vs.glsl"}; 		// vertex shader of solver program
	// const char* fragment_path_fbo = {"shaders_FDTD/fbo_fs.glsl"}; 		// fragment shader of solver program
	// const char* vertex_path_render = {"shaders_FDTD/render_vs.glsl"};	// vertex shader of render program
	// const char* fragment_path_render = {"shaders_FDTD/render_fs.glsl"};	// fragment shader of render program
	
	// // FBO shader program (solver)
	// GLuint shader_program_fbo = 0;
	// GLuint vs_fbo = 0;
	// GLuint fs_fbo = 0;
	// if(!loadShaderProgram(vertex_path_fbo, fragment_path_fbo, vs_fbo, fs_fbo, shader_program_fbo))
	// 	return 1;

	// // screen shader program (render)
	// GLuint shader_program_render = 0;
	// GLuint vs_render = 0;
	// GLuint fs_render = 0;
	// if(!loadShaderProgram(vertex_path_render, fragment_path_render, vs_render, fs_render, shader_program_render))
	// 	return 1;

	return 0;
}