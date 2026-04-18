cmake_minimum_required(VERSION 3.16)
project(yolo_onnx)

include(FetchContent)

FetchContent_Declare(
  onnxruntime
  URL https://github.com/microsoft/onnxruntime/releases/download/v1.19.0/onnxruntime-linux-x64-gpu-1.19.0.tgz
)
FetchContent_MakeAvailable(onnxruntime)

find_package(OpenCV REQUIRED)

fins_add_node(${PROJECT_NAME} nodes.cpp)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${onnxruntime_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${OpenCV_LIBS}
    ${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so
)