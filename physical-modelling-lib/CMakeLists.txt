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
# include glfw3
add_subdirectory(${PROJECT_SOURCE_DIR}/includes/glfw)
set(GLFW_SRC PUBLIC ${PROJECT_SOURCE_DIR}/includes/glfw/src)
set(GLFW_INCLUDE PUBLIC ${PROJECT_SOURCE_DIR}/includes/glfw/include)
# include glew
add_subdirectory(${PROJECT_SOURCE_DIR}/includes/glew/build/cmake)
set(GLEW_SRC PUBLIC ${PROJECT_SOURCE_DIR}/includes/glew/src)
set(GLEW_INCLUDE PUBLIC ${PROJECT_SOURCE_DIR}/includes/glew/include)

# configure pybind
add_subdirectory(${PROJECT_SOURCE_DIR}/includes/pybind11)
pybind11_add_module(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/src/py.cpp)
# link libraries to pybind
target_link_libraries(${PROJECT_NAME} PUBLIC glfw glew)
target_link_directories(${PROJECT_NAME} ${GLFW_SRC} ${GLEW_SRC})
target_include_directories(${PROJECT_NAME} ${GLFW_INCLUDE} ${GLEW_INCLUDE})
if(APPLE)
	target_link_libraries(${PROJECT_NAME} PUBLIC "-framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo")
endif()

# compile standalone app
add_executable(physical_modelling_lib ${PROJECT_SOURCE_DIR}/src/main.cpp)
# link libraries to executable 
target_link_libraries(physical_modelling_lib PUBLIC glfw glew)
target_link_directories(physical_modelling_lib ${GLFW_SRC} ${GLEW_SRC})
target_include_directories(physical_modelling_lib ${GLFW_INCLUDE} ${GLEW_INCLUDE})
if(APPLE)
	target_link_libraries(physical_modelling_lib PUBLIC "-framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo")
endif()