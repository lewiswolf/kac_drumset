cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 20)
include(FetchContent)

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

# include pybind as a remote dependency
FetchContent_Declare(
	pybind11
	GIT_REPOSITORY https://github.com/pybind/pybind11
)
FetchContent_MakeAvailable(pybind11)

# include kac_core
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/kac_core)

# include src files
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/kac_drumset/externals")

# build modules
pybind11_add_module(_geometry MODULE ${CMAKE_CURRENT_SOURCE_DIR}/kac_drumset/externals/_geometry.cpp)
pybind11_add_module(_physics MODULE ${CMAKE_CURRENT_SOURCE_DIR}/kac_drumset/externals/_physics.cpp)
target_link_libraries(_geometry PRIVATE kac_core)
target_link_libraries(_physics PRIVATE kac_core)
install(TARGETS _geometry _physics DESTINATION .)

# manually copy after build
add_custom_command(TARGET _geometry POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_geometry> ${CMAKE_CURRENT_SOURCE_DIR}/kac_drumset/externals
)
add_custom_command(TARGET _physics POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_physics> ${CMAKE_CURRENT_SOURCE_DIR}/kac_drumset/externals
)