cmake_minimum_required(VERSION 3.27)
project(kac_drumset LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
include(FetchContent)

# include pybind as a remote dependency
FetchContent_Declare(
	pybind11
	GIT_REPOSITORY https://github.com/pybind/pybind11
)
FetchContent_MakeAvailable(pybind11)

# include kac_core
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/kac_core)

# build modules
pybind11_add_module(_geometry MODULE ${CMAKE_CURRENT_SOURCE_DIR}/kac_drumset/externals/_geometry.cpp)
pybind11_add_module(_physics MODULE ${CMAKE_CURRENT_SOURCE_DIR}/kac_drumset/externals/_physics.cpp)
target_link_libraries(_geometry PRIVATE kac_core)
target_link_libraries(_physics PRIVATE kac_core)
install(
    TARGETS _geometry _physics
    LIBRARY DESTINATION kac_drumset/externals
    RUNTIME DESTINATION kac_drumset/externals
	ARCHIVE DESTINATION kac_drumset/externals
)
