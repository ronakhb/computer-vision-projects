/*

functions.hpp
Project 4

Created by Ronak Bhanushali and Rucha Pendharkar on 2/8/24

This file contains function definitions used for multiple tasks.


*/


#include <iostream>
#include <stdio.h>
#include <opencv2/aruco.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/opencv.hpp>

/**
 * @brief Get the aruco targets from image
 * 
 * @param src - input image
 * @return std::vector<std::vector<cv::Point2f>> corners of targets detected
 */

std::pair<std::vector<int> ,std::vector<std::vector<cv::Point2f>>> getTarget(cv::Mat& src, cv::Mat& detected_frame);

/**
 * @brief Load calibration parameters from file
 * 
 * @param file_path - xml or yaml file containing parameters
 * @param camera_matrix - variable to hold camera_matrix data
 * @param dist_coeffs - variable to hold dist_coefficient data
 * @return int - return 0 for success
 */

int loadParameters(const std::string& file_path, cv::Mat& camera_matrix, cv::Mat& dist_coeffs);

/**
 * @brief Undistort a given image frame
 * 
 * @param input_image -  input frame
 * @param camera_matrix - variable containing camera_matrix data 
 * @param dist_coeffs - variable containing dist_coefficient data
 * @param undistorted_image - undistorted frame
 * @return int - return 0 for success
 */

int undistortImage(const cv::Mat& srcs, const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs, cv::Mat& undistorted_image);


/**
 * @brief Generates a grid of aruco markers
 * 
 * @param num_markers - number of markers
 * @param spacing - spacing between markers in pixels
 * @return int 0 for successs
 */

int generateArUcoMarkers(int num_markers, int marker_size, int spacing);
