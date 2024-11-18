/*
Ronak Bhanushali
CS 5330
Spring 2024

Project 1
*/

https://github.com/ronakhb/cs5330.git - Please let me know if access is needed

NOTE: I enrolled in the class late which is why this submission is a week late. Professor allowed me to submit the extensions late as a result of this

Project Structure
Directory Overview

All header files can be found in include folder
All cpp files can be found in the src folder
main.cpp is used to make the executable and has the main loop in it

build_project.sh: This script is used to build the project, facilitating compilation and executable generation.
setup_project.sh: This script assists in setting up the project, handling any necessary configurations or dependencies.

Usage
Key Usage
'q': Quit the application.
's': Save the current frame.
'g' to 'n': Toggle between different video processing modes.
'g': Grayscale
'h': Custom Greyscale
'p': Sepia
'b': Blur 5x5
'x': Sobel X 3x3
'y': Sobel Y 3x3
'm': Magnitude Sobel
'i': Blur Quantize
'f': Face Detection
'c': Cartoonize
'a': Face Blur
'n': Negative Image
'v': Toggle vignette effect.
'r': Toggle video recording.
Build
To build the project, run the following scripts in the scripts folder:

./scripts/setup_project.sh


./scripts/build_project.sh


The project should be built and executables should be in the bin folder

To run the main project executable for task1:

./bin/project1 task1 <path to any image file>

To run the main project executable for all other tasks:

./bin/project1 task2

To run the timeBlur comparison run,

./bin/time_test


Link to recording demo - https://drive.google.com/file/d/1Wo1bsFBOb8uMHiP9HrlIJpJljbPWNlRm/view?usp=sharing