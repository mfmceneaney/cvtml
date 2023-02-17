cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

find_package(TorchSparse REQUIRED)
find_package(TorchScatter REQUIRED)
#find_package(TorchCluster REQUIRED)

message("DEBUGGING: TORCH_FOUND = ${TORCH_FOUND}")#DEBUGGING
message("DEBUGGING: TORCH_LIBRARIES = ${TORCH_LIBRARIES}")#DEBUGGING
message("DEBUGGING: TORCH_CXX_FLAGS = ${TORCH_CXX_FLAGS}")#DEBUGGING

add_executable(example-app example-app.cpp)

target_link_libraries(example-app "${TORCH_LIBRARIES}")

target_link_libraries(example-app TorchSparse::TorchSparse)
target_link_libraries(example-app TorchScatter::TorchScatter)
#target_link_libraries(example-app TorchCluster::TorchCluster)

set_property(TARGET example-app PROPERTY CXX_STANDARD 14)
