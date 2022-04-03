cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 11)

project(kac_drumset)

# include geometry-cpp
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/lib/geometry-cpp/src")
file(GLOB GEOMETRY "${CMAKE_CURRENT_SOURCE_DIR}/lib/geometry-cpp/src/geometry.hpp")

# include pybind as a remote dependency
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
)
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

# Scikit-Build does not add your site-packages to the search path
# automatically, so we need to add it _or_ the pybind11 specific directory
# here.
if(SKBUILD)
	execute_process(
    	COMMAND "${PYTHON_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    	OUTPUT_VARIABLE _tmp_dir
    	OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT
	)
	list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
endif()

# build modules
pybind11_add_module(_geometry MODULE ${CMAKE_CURRENT_SOURCE_DIR}/kac_drumset/geometry/_geometry.cpp)
install(TARGETS _geometry DESTINATION .)