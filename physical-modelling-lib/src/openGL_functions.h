#pragma once
#ifndef OPENGL_FUNCTIONS
#define OPENGL_FUNCTIONS
	// core
	#include <string>
	// includes
	#include <GL/glew.h>    	// OpenGL
	#include <GLFW/glfw3.h>		// OpenGL Window

	GLFWwindow *initOpenGL(int width, int heigth, std::string windowName, float magnifier);

#endif