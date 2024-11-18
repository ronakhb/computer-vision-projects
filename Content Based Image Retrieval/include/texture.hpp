/*
Ronak Bhanushali
CS 5330
Spring 2024

This file applies filters to an image
*/

#include <iostream>
#include <stdio.h>
#include "opencv2/opencv.hpp"

/**
 * @brief Applies a 3x3 Sobel filter in the X direction to the source image.
 *
 * @param src Source cv::Mat.
 * @param dst Destination cv::Mat.
 * @return int
 */
int sobelX3x3(cv::Mat &src, cv::Mat &dst);

/**
 * @brief Applies a 3x3 Sobel filter in the Y direction to the source image.
 *
 * @param src Source cv::Mat.
 * @param dst Destination cv::Mat.
 * @return int
 */
int sobelY3x3(cv::Mat &src, cv::Mat &dst);

/**
 * @brief Calculates the gradient magnitude image using Sobel operators.
 *
 * This function computes the gradient magnitude of an image using Sobel operators
 * along the x and y directions. The resulting magnitude is stored in the destination
 * matrix.
 *
 * @param sx Sobel x matrix
 * @param sy Sobel y matrix
 * @param dst Destination cv::Mat to store the gradient magnitude image.
 * @return int Returns 0 on success.
 */
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

int textureMatching(char* argv[]);
