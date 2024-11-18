# CMakeLists.txt in the src directory

# Define the source files for the library
set(PROJECT_SOURCES
    projects/project1/imgDisplay.cpp
    projects/project1/vidDisplay.cpp
    projects/project1/filters.cpp
    projects/project1/faceDetect.cpp
    projects/project1/showFaces.cpp
    # Add other source files as needed
)

include_directories(${CMAKE_SOURCE_DIR}/include)
message("CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
# Create a library from the source files
add_library(project1_lib ${PROJECT_SOURCES})
target_link_libraries(project1_lib ${OpenCV_LIBS})