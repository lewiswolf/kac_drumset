cmake_minimum_required(VERSION 3.20.3)
project(physical-lib)
add_subdirectory(pybind11)
pybind11_add_module(physical_lib main.cpp)