/*
Ronak Bhanushali
CS 5330
Spring 2024

Project 1
*/

#include <iostream>
#include <stdio.h>
#include "opencv2/opencv.hpp"


/**
 * @brief Function to toggle between window and Fullscreen
 * 
 * @param img image to display
 */

void toggleFullScreen(cv::Mat img);

/**
 * @brief Function to display image and monitor keypress
 * 
 * @param path_to_img Path to image
 */
int displayImage(std::string path_to_img = "/home/ronak/Downloads/cat.jpg");