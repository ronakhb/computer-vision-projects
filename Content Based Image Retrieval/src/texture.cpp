/*
Prem Sukhadwala and Ronak Bhanushali
January 2024
CPP file for the functions being used in the textureMatching file
*/

#include <iostream> //standard input output
#include <dirent.h>
#include "texture.hpp"

//Implements a 3x3 sobel Y filter
int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
    int num_rows = src.rows;
    int num_cols = src.cols;
    cv::Mat temp_mat_1;
    src.convertTo(temp_mat_1,CV_32S);
    cv::Mat temp_mat_2 = temp_mat_1.clone();
    cv::Mat temp_mat_3 = temp_mat_2.clone();
    int kernel1[] = { 1, 2, 1 };
	int kernel2[] = { -1, 0, 1 };

//Vertical Kernel

	for (int i = 1; i < num_rows - 1; i++)
	{
		for (int j = 1; j < num_cols - 1; j++)
        {
            cv::Vec3i pixel_value = cv::Vec3i(0, 0, 0);
            for (int k = 0; k < 3; k++)
            {
                pixel_value += kernel1[k] * temp_mat_1.at<cv::Vec3i>(i,j+k-1);
            }
            temp_mat_2.at<cv::Vec3i>(i, j) = pixel_value;
        }
    }

//Horizontal Kernel

	for (int i = 1; i < num_rows - 1; i++)
	{
		for (int j = 1; j < num_cols - 1; j++)
        {
            cv::Vec3i pixel_value = cv::Vec3i(0, 0, 0);
            for (int k = 0; k < 3; k++)
            {
                pixel_value += kernel2[k] * temp_mat_2.at<cv::Vec3i>(i+k-1,j);
            }
            temp_mat_3.at<cv::Vec3i>(i, j) = pixel_value;
        }
    }
    cv::convertScaleAbs(temp_mat_3,dst,1); //Take absolute value of pixels since some might be negative. Note that scaling factor I used is 1 but can 0.25 can be used too. that would dim the edges a bit
	dst.convertTo(dst,CV_8UC3);
	return 0;
}

//Implements a 3x3 sobel X filter
int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    int num_rows = src.rows;
    int num_cols = src.cols;

    cv::Mat temp_mat_1; //temporary matrix to clone src before processing

    src.convertTo(temp_mat_1,CV_32S);

    cv::Mat temp_mat_2 = temp_mat_1.clone(); //temporary matrix to store image after first convolution

    cv::Mat temp_mat_3 = temp_mat_2.clone(); //temporary matrix to store image after second convolution

    int kernel1[] = { -1, 0, 1 };
	int kernel2[] = { 1, 2, 1 };

//Vertical Kernel

	for (int i = 1; i < num_rows - 1; i++)
	{
		for (int j = 1; j < num_cols - 1; j++)
        {
            cv::Vec3i pixel_value = cv::Vec3i(0, 0, 0);
            for (int k = 0; k < 3; k++)
            {
                pixel_value += kernel1[k] * temp_mat_1.at<cv::Vec3i>(i,j+k-1);
            }
            temp_mat_2.at<cv::Vec3i>(i, j) = pixel_value;
        }
    }

//Horizontal Kernel

	for (int i = 1; i < num_rows - 1; i++)
	{
		for (int j = 1; j < num_cols - 1; j++)
        {
            cv::Vec3i pixel_value = cv::Vec3i(0, 0, 0);
            for (int k = 0; k < 3; k++)
            {
                pixel_value += kernel2[k] * temp_mat_2.at<cv::Vec3i>(i+k-1,j);
            }
            temp_mat_3.at<cv::Vec3i>(i, j) = pixel_value;
        }
    }
    cv::convertScaleAbs(temp_mat_3,dst,1); //Take absolute value of pixels since some might be negative. Note that scaling factor I used is 1 but can 0.25 can be used too. that would dim the edges a bi
	dst.convertTo(dst,CV_8UC3);
	return 0;
}

int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst)
{
    int num_rows = sx.rows; //numver of rows
    int num_cols = sy.cols; //number of columns
    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_cols; j++)
        {
            cv::Vec3b x = sx.at<cv::Vec3b>(i, j); //get x gradient at pixel
            cv::Vec3b y = sy.at<cv::Vec3b>(i, j); //get y gradient at pixel

            for (int c = 0; c < 3; c++)
            {
                dst.at<cv::Vec3b>(i, j)[c] = sqrtf(pow(x[c], 2) + pow(y[c], 2)); //calculate gradient magnitude
            }
        }
    }
    return 0;
}