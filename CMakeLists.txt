cmake_minimum_required(VERSION 3.22)

project(trt_plugin_kit LANGUAGES CXX CUDA)

# Compiler standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wno-deprecated-declarations -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -Wall -Xcompiler -fPIC")
set(CMAKE_CUDA_ARCHITECTURES 86)

# Path to custom CMake modules
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Find TensorRT and CUDA Toolkit
find_package(TensorRT REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Include base plugin utilities
include_directories(base)

# Add each plugin
file(GLOB PLUGIN_DIRS CONFIGURE_DEPENDS plugins/*)
foreach(PLUGIN_PATH ${PLUGIN_DIRS})
    if(IS_DIRECTORY ${PLUGIN_PATH})
        add_subdirectory(${PLUGIN_PATH})
    endif()
endforeach()