cmake_minimum_required(VERSION 3.20.3)
project(physical-lib)

# Add source files
file(GLOB_RECURSE SOURCE_FILES 
	${CMAKE_SOURCE_DIR}/src/*.c
	${CMAKE_SOURCE_DIR}/src/*.cpp)
	
# Add header files
file(GLOB_RECURSE HEADER_FILES 
	${CMAKE_SOURCE_DIR}/*.h
	${CMAKE_SOURCE_DIR}/*.hpp)

# python bindings
add_subdirectory(includes/pybind11)
pybind11_add_module(physical_lib src/main.cpp)