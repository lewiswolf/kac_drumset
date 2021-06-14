// core
#include <string>
// includes
#include <GL/glew.h>    	// OpenGL
#include <GLFW/glfw3.h>		// OpenGL Window
// src
#include "openGL_functions.h"

using namespace std;

// error handling for GLFW
void glfwErrorCallback(int val, const char *name) {
	string error = "GLFW init error: " + to_string(val);
	error += "\n\t- ";
	error += name;
	error += "\n";
	perror(error.c_str());
}

GLFWwindow *initOpenGL(int width, int heigth, string windowName, float magnifier) {
	//**********************************************************************************************

	// let's set an error callback, just in case...
	glfwSetErrorCallback(glfwErrorCallback);

	// start GL context and O/S window using the GLFW helper library
	if (!glfwInit()) {
		fprintf (stderr, "ERROR: could not start GLFW3\n");
		return NULL;
	}

	// create window handler
	GLFWwindow* window = glfwCreateWindow (width*magnifier, heigth*magnifier, windowName.c_str(), NULL, NULL);

	if (!window) {
		fprintf (stderr, "ERROR: could not open window with GLFW3\n");
		glfwTerminate();
		return NULL;
	}

	glfwMakeContextCurrent(window);

	// start GLEW extension handler
	glewExperimental = GL_TRUE;
	glewInit ();

	// print version info
	const GLubyte* renderer = glGetString(GL_RENDERER); // get renderer string
	const GLubyte* version  = glGetString(GL_VERSION);  // version as a string
	printf("Renderer: %s\n", renderer);
	printf("OpenGL version supported %s\n", version);

	// this disables the VSync with the monitor! FUNDAMENTAL!
	glfwSwapInterval(0);

	return window; // (:
}
