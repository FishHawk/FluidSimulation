cmake_minimum_required(VERSION 3.15)
project(FluidSimulation
    VERSION 1.0
    LANGUAGES C CXX CUDA)

add_subdirectory(lib)
add_subdirectory(src)