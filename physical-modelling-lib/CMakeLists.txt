cmake_minimum_required(VERSION 3.20.3)
project(physical_lib)

# configure compiler
set(CMAKE_CXX_STANDARD 17)

# include src
file(
	GLOB_RECURSE
	${PROJECT_SOURCE_DIR}/src/*.cpp
)
file(
	GLOB_RECURSE
	${PROJECT_SOURCE_DIR}/src/*.hpp
)

# configure pybind
add_subdirectory(${PROJECT_SOURCE_DIR}/includes/pybind11)
pybind11_add_module(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/src/main.cpp)

# include glfw3
set(GLFW_SRC PUBLIC ${PROJECT_SOURCE_DIR}/includes/glfw/src)
set(GLFW_INCLUDE PUBLIC ${PROJECT_SOURCE_DIR}/includes/glfw/include)

# include glew
set(GLEW_SRC PUBLIC ${PROJECT_SOURCE_DIR}/includes/glew/auto/src)
set(GLEW_INCLUDE PUBLIC ${PROJECT_SOURCE_DIR}/includes/glew/include)

# link libraries
target_link_directories(${PROJECT_NAME} ${GLFW_SRC} ${GLEW_SRC})
target_include_directories(${PROJECT_NAME} ${GLFW_INCLUDE} ${GLEW_INCLUDE})
if(APPLE)
	# macos specifics
	target_link_libraries(${PROJECT_NAME} PUBLIC "-framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo")
endif()