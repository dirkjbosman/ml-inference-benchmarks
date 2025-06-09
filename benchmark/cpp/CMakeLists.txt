cmake_minimum_required(VERSION 3.10)
project(cpp_benchmark)

add_executable(cpp_benchmark main.cpp)

target_include_directories(cpp_benchmark PRIVATE ${ONNXRUNTIME_DIR}/include)
target_link_directories(cpp_benchmark PRIVATE ${ONNXRUNTIME_DIR}/lib)
target_link_libraries(cpp_benchmark PRIVATE onnxruntime)
