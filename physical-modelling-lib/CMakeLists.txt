cmake_minimum_required(VERSION 3.20.3)
project(physical_lib)

# configure compiler
set(CMAKE_CXX_STANDARD 17)

# include src
file(
	GLOB_RECURSE
	${CMAKE_SOURCE_DIR}/src/*.cpp
)
file(
	GLOB_RECURSE
	${CMAKE_SOURCE_DIR}/*.hpp
)

# indluce dependencies
add_subdirectory(includes/glfw)

add_subdirectory(includes/pybind11)
pybind11_add_module(physical_lib src/main.cpp)