#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

int applyGaborFilter(cv::Mat& input_image, cv::Mat& filtered_image, int width, int height, double sigma, double theta, double lambda, double gamma, double psi);
