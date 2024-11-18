# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

project(Project1)
find_package(OpenCV REQUIRED)

# Add the path to your header files
include_directories(include)
include_directories(include/projects/project1)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Add the path to your source files
add_subdirectory(src)

# Add the executable target
add_executable(project1 src/projects/project1/main.cpp)
add_executable(time_test src/projects/project1/timeBlur.cpp)

# Link your executable with the source files
target_link_libraries(project1 project1_lib)
target_link_libraries(time_test project1_lib)
# Check for the presence of DEBUG file