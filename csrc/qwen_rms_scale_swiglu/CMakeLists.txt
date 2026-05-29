cmake_minimum_required(VERSION 3.18)
project(qwen_rms_scale_swiglu_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS
    /usr/include
    /usr/include/x86_64-linux-gnu
    /usr/local/include
    /usr/local/tensorrt/include
    /usr/local/TensorRT/include
)

find_library(NVINFER_LIB nvinfer
    PATHS
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    /usr/local/lib
    /usr/local/tensorrt/lib
    /usr/local/TensorRT/lib
)

if(NOT TENSORRT_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find TensorRT NvInfer.h. Set TENSORRT_INCLUDE_DIR manually if needed.")
endif()

if(NOT NVINFER_LIB)
    message(FATAL_ERROR "Could not find TensorRT libnvinfer. Set NVINFER_LIB manually if needed.")
endif()

add_library(qwen_rms_scale_swiglu_plugin SHARED
    plugin/qwenRmsScaleSwigluPlugin.cpp
    kernel/qwenRmsScaleSwigluKernel.cu
)

target_include_directories(qwen_rms_scale_swiglu_plugin PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(qwen_rms_scale_swiglu_plugin PRIVATE
    CUDA::cudart
    ${NVINFER_LIB}
)

target_compile_options(qwen_rms_scale_swiglu_plugin PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
)

set_target_properties(qwen_rms_scale_swiglu_plugin PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    OUTPUT_NAME qwen_rms_scale_swiglu_plugin
)
