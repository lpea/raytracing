cmake_minimum_required(VERSION 3.24)
project(RayTracingCpp)

find_package(OpenCV COMPONENTS core imgproc highgui)

add_executable(raytracer main.cpp)
target_link_libraries(raytracer opencv_core opencv_highgui)
