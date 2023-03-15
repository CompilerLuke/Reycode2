cmake_minimum_required(VERSION 3.24)
project(Reycode)

set(REY_PLATFORM_CUDA off)

add_subdirectory(reycode)
add_subdirectory(reycode_meshing)
add_subdirectory(reycode_solver)
add_subdirectory(reycode_graphics)
add_subdirectory(reycode_viewer)