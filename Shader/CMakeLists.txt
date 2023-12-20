cmake_minimum_required(VERSION 3.24)
project(RayTracingShader)

find_package(OGRE 14.1.2 REQUIRED COMPONENTS Bites CONFIG)

add_executable(shader_test main.cpp)
target_include_directories(shader_test PRIVATE .)
target_link_libraries(shader_test OgreBites)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/resources.cfg"
  "${CMAKE_CURRENT_BINARY_DIR}")
