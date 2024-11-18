/*
Prem Sukhadwala and Ronak Bhanushali
January 2024
Include file for the functions in functions.cpp
*/

#ifndef CVS_UTIL_H
#define CVS_UTIL_H

#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

/**
 * @brief Function to write color features
 * to a CSV file
 * @param argv[] command line arguments
 * @return int 0 for successful
 */
int writeFeaturesToCSV(char *argv[]);

/**
 * @brief Function to fetch color features from
 * the target image
 * @param filePath target image path as a command line argument
 * @return int 0 for successful
 */
std::vector<float> targetColorFeatures(char *filePath);

/**
 * @brief Function to compare color features of
 * the target image and the images in the set by
 * reading them from a CSV file
 * @param argv[] command line arguments
 * @return taget image vector
 */
int compareFeatures(char *argv[]);

/**
 * @brief Function to compare ResNet18 512 feature vector of
 * the target image and the images in the set by
 * reading them from a CSV file
 * @param argv[] command line arguments
 * @return int 0 for successful
 */
int compareEmbeddedVectors(char *argv[]);

/**
 * @brief Function to fetch color features for 
 * all the images in a directory and write
 * them to a CSV file
 * @param csv_file_name CSV file path
 * @param filePath image file path
 * @return int 0 for successful
 */
int getColorFeatures(char *csv_file_name,char *filePath);

/**
 * @brief Function to write image data (path and features) to CSV file
 * @param csv_file_name CSV file path
 * @param image_filename image file path
 * @param image_data vector of image data
 * @return int 0 for successful
 */
int append_image_data_csv( char *csv_file_name, char *image_filename, std::vector<float> &image_data, int reset_file );

/**
 * @brief Function to read float data from a CSV file
 * @param fp CSV file path
 * @param v parameter where data is stored
 * @return int 0 for successful
 */
int getfloat(FILE *fp, float *v);

/**
 * @brief Function to read integer data from a CSV file
 * @param fp CSV file path
 * @param v parameter where data is stored
 * @return int 0 for successful
 */
int getint(FILE *fp, int *v);

/**
 * @brief Function to read string data from a CSV file
 * @param fp CSV file path
 * @param os array parameter where data is stored
 * @return int 0 for successful
 */
int getstring( FILE *fp, char os[] );

/**
 * @brief Function to read image data (path and features) from CSV file
 * @param filename CSV file path
 * @param filenames vector where the image paths are stored
 * @param data vector of vectors wehre the image feature data is stored
 * @return int 0 for successful
 */
int read_image_data_csv( char *filename, std::vector<char *> &filenames, std::vector<std::vector<float>> &data, int echo_file );

/**
 * @brief Function to match histograms using histogram intersection
 * @param argv command line arguments
 * @return int 0 for successful
 */
int histogramMatching(char* argv[]);

/**
 * @brief Function to calculate RG histogram
 * @param img given image
 * * @param hist generated histogram
 * @return int 0 for successful
 */
int generateHistogramRG(cv::Mat &img, cv::Mat &hist);

/**
 * @brief Function to match two histograms using histogram intersection and weighted sum
 * @param argv command line arguments
 * @return int 0 for successful
 */
int multiHistogramMatching(char* argv[]);

int customHistMatching(char* argv[]);

/**
 * @brief Function to calculate histogram intersection area
 * @param hist1 histogram 1
 * @param hist2 histogram 2
 * @return total intersection area
 */
float histIntersectionArea(cv::Mat &hist1, cv::Mat &hist2);

#endif