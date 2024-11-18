cmake_minimum_required(VERSION 3.0.0)
project(project4 VERSION 0.1.0)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)

# Find OpenCV
find_package(OpenCV REQUIRED)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
option(OPENCV_ENABLE_NONFREE "Enable non-free algorithms" ON)

# Include directories for OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

# Add the executable and link against OpenCV and Boost libraries
add_executable(CameraCalib src/calibration.cpp)
target_link_libraries(CameraCalib ${OpenCV_LIBS})

add_executable(task4 src/task4.cpp src/functions.cpp include/functions.hpp)
target_link_libraries(task4 ${OpenCV_LIBS})

add_executable(task5 src/task5.cpp src/functions.cpp include/functions.hpp)
target_link_libraries(task5 ${OpenCV_LIBS})

add_executable(task6 src/task6.cpp src/functions.cpp include/functions.hpp)
target_link_libraries(task6 ${OpenCV_LIBS})

add_executable(task7 src/task7.cpp)
target_link_libraries(task7 ${OpenCV_LIBS})
