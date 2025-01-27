cmake_minimum_required(VERSION 3.10)
project(CustomOpProject)

# Set ONNX Runtime directories (no need for CACHE if paths are fixed)
set(ONNXRUNTIME_BIN_DIR "/usr/local/bin")
set(ONNXRUNTIME_INCLUDE_DIR "/usr/local/include")
set(ONNXRUNTIME_LIB_DIR "/usr/local/lib")  # Adjust this if libraries are elsewhere

# Include ONNX Runtime headers
include_directories(${ONNXRUNTIME_INCLUDE_DIR})

# Link ONNX Runtime library
link_directories(${ONNXRUNTIME_LIB_DIR})

# Build custom operator
add_library(custom_op SHARED src/custom_operator.cc)

# Link with ONNX Runtime
target_link_libraries(custom_op PRIVATE onnxruntime)
