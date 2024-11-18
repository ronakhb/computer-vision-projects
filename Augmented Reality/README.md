# Project IV : Calibration and Augmented Reality

## Created by Ronak Bhanushali and Rucha Pendharkar on 3/18/24

The goal of this project is to understand and perform camera calibration. The calibration parameters are then used to create a virtual object. The result is a program that is able to recognize a target and place a virtual object relative to the target, even if the target or camera is rotated or translated

## Video Link

Link to containing the virtual object we created - https://www.youtube.com/shorts/YXEigihUHyo?feature=share
## Features

- **Camera Calibration:** : The system captures video in real-time using the camera from an input can be mobile phone, laptop webcam and performs camera calibration. Run the calibration.cpp file for tasks 1-3

- **Virtual Object**: The virtual object is created using files task6.cpp. Run files task4.cpp and task5.cpp to calculate current position of the camera and project the corners outside the 3D axes. 

- **Feature Detection**: Regions are identified using OpenCV's SURF function. Run task7.cpp for the same. 
 
## Usage

- Press 'q' to exit the application.
- Press 's' to save images in calibration.cpp file.

## Environment 
The scripts were authored using VS Code, and code compilation took place in the Ubuntu 20.04.06 LTS environment, utilizing CMake through the terminal.
