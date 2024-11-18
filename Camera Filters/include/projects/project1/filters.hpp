/*
Ronak Bhanushali
CS 5330
Spring 2024

This file applies filters to an image
*/

#include <iostream>
#include <stdio.h>
#include "opencv2/opencv.hpp"

enum VideoMode {
    GRAYSCALE = 1,
    GREYSCALE_CUSTOM = 2,
    SEPIA = 3,
    BLUR_5X5 = 4,
    SOBEL_X_3X3 = 5,
    SOBEL_Y_3X3 = 6,
    MAGNITUDE_SOBEL = 7,
    BLUR_QUANTIZE = 8,
    FACE_DETECTION = 9,
    CARTOONIZE = 10,
    FACE_BLUR = 11,
    NEGATIVE_IMAGE = 12
};

/**
 * @brief Custom greyscale function
 * 
 * @param src source cv::Mat
 * @param dst destination cv::Mat
 * @return int 0 for successful
 */

int greyscale(cv::Mat &src, cv::Mat &dst);

/**
 * @brief Applies sepia filter to given image
 * 
 * @param src  source cv::Mat
 * @param dst  destination cv::Mat
 * @return int 0 for successful
 */
int sepia(cv::Mat &src, cv::Mat &dst);

/**
 * @brief Applies sepia filter to given image using direct multiplication with a kernel instead of element wise operation
 * 
 * @param src  source cv::Mat
 * @param dst  destination cv::Mat
 * @return int 0 for successful
 */
int sepia2(cv::Mat &src, cv::Mat &dst);

/**
 * @brief Applies Vignetting effect to any image
 * 
 * @param src  source cv::Mat
 * @param dst  destination cv::Mat
 * @return int 0 for successful
 */
int vignette(cv::Mat &src, cv::Mat &dst);

/**
 * @brief applies a 5x5 gaussian blur to src image
 * 
 * @param src source cv::Mat
 * @param dst destination cv::Mat
 * @return int 
 */
int blur5x5_1( cv::Mat &src, cv::Mat &dst );

/**
 * @brief An alternative implementation of a 5x5 Gaussian blur on the source image. Multiplies kernel directly
 *
 * @param src Source cv::Mat.
 * @param dst Destination cv::Mat.
 * @return int
 */
int blur5x5_1_alt(cv::Mat &src, cv::Mat &dst);

/**
 * @brief Applies a 5x5 Gaussian blur to the source image using split kernel
 *
 * @param src Source cv::Mat.
 * @param dst Destination cv::Mat.
 * @return int
 */
int blur5x5_2(cv::Mat &src, cv::Mat &dst);

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

/**
 * @brief Applies blur and quantization to the source image.
 *
 * This function applies blur and quantization to the source image, producing
 * a blurred and quantized version in the destination matrix.
 *
 * @param src Source cv::Mat
 * @param dst Destination cv::Mat for the blurred and quantized image.
 * @param levels Number of quantization levels.
 * @return int Returns 0 on success.
 */
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);

/**
 * @brief Applies cartoon effect to the source image.
 *
 * This function applies a cartoon effect to the source image by combining
 * blur and quantization. The resulting image is stored in the destination matrix.
 *
 * @param src Source cv::Mat
 * @param dst Destination cv::Mat for the cartoon effect.
 * @param levels Number of quantization levels.
 * @param threshold Threshold for edge detection.
 * @return int Returns 0 on success.
 */
int cartoon(cv::Mat &src, cv::Mat &dst, int levels, int threshold);

/**
 * @brief Applies blur and replaces detected faces in the input frame.
 *
 * This function applies blur to the entire image and replaces the detected faces
 * with the original blurred image. The result is stored in the destination matrix.
 *
 * @param frame Input cv::Mat containing the original frame.
 * @param dst Destination cv::Mat for the modified frame.
 * @param faces Vector of cv::Rect containing the coordinates of detected faces.
 * @return int Returns 0 on success.
 */
int applyBlurAndReplaceFace(cv::Mat& frame, cv::Mat& dst, const std::vector<cv::Rect>& faces);

/**
 * @brief Creates a negative image from the input frame.
 *
 * This function generates a negative image from the input frame, where the
 * colors are inverted. The resulting image is stored in the destination matrix.
 *
 * @param src Input cv::Mat containing the original frame.
 * @param dst Destination cv::Mat for the negative image.
 * @return int Returns 0 on success.
 */
int negativeImage(cv::Mat& src, cv::Mat& dst);