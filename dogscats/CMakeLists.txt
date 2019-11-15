cmake_minimum_required(VERSION 3.1)
project(dogscats)

find_package(Torch REQUIRED)
find_package(OpenCV 4.1.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(dogscats main.cpp)
target_link_libraries(dogscats ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET dogscats PROPERTY CXX_STANDARD 11)