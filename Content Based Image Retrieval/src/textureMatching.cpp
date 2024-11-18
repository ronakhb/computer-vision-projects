/*
Prem Sukhadwala and Ronak Bhanushali
January 2024
Code for Content Based Image Retrieval using texture and color features
*/

#include<iostream>
#include "functions.hpp"
#include "texture.hpp"
#include <dirent.h>

int textureMatching(char *argv[])
{
    char dirname[256];
    char buffer[256];
    FILE *fp;
    DIR *dirp;
    struct dirent *dp;

    int numFiles = 0;
    char *filePaths[1600];

    // get the directory path
    strcpy(dirname, argv[1]);
    printf("Processing directory %s\n", dirname);
    // open the directory
    dirp = opendir(dirname);
    if (dirp == NULL)
    {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }

    // loop over all the files in the image file listing
    while ((dp = readdir(dirp)) != NULL)
    {

        // check if the file is an image
        if (strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif"))
        {
            printf("processing image file: %s\n", dp->d_name);
            // build the overall filename
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);
            printf("full path name: %s\n", buffer);
            filePaths[numFiles] = strdup(buffer);
            numFiles++;
        }
    }
    char target_image_path[256];
    strcpy(target_image_path, argv[2]);

    std::vector<std::pair<float, std::string>> image_ranks;

    cv::Mat target_image = cv::imread(target_image_path);
    cv::Mat sx;
    cv::Mat sy;
    cv::Mat target_magnitude = target_image.clone();
    sobelX3x3(target_image,sx);
    sobelY3x3(target_image,sy);
    magnitude(sx, sy ,target_magnitude);

    cv::Mat target_histogram_sobel;
    cv::Mat target_histogram_color;
    generateHistogramRG(target_image, target_histogram_color);
    generateHistogramRG(target_magnitude, target_histogram_sobel);

    for (int i = 0; i < numFiles; i++)
    {
        cv::Mat query_image = cv::imread(filePaths[i]);
        cv::Mat query_sx;
        cv::Mat query_sy;
        cv::Mat query_magnitude = query_image.clone();
        sobelX3x3(query_image, query_sx);
        sobelY3x3(query_image, query_sy);
        magnitude(query_sx, query_sy, query_magnitude);
        cv::Mat query_histogram_color;
        cv::Mat query_histogram_sobel;
        generateHistogramRG(query_image, query_histogram_color);
        generateHistogramRG(query_magnitude, query_histogram_sobel);
        float overlapping_area_color = histIntersectionArea(target_histogram_color, query_histogram_color);
        float overlapping_area_sobel = histIntersectionArea(target_histogram_sobel, query_histogram_sobel);
        float weighted_area_sum = 0.5 * overlapping_area_color + 0.5 * overlapping_area_sobel;
        image_ranks.push_back(std::make_pair(weighted_area_sum, filePaths[i]));
    }
    sort(image_ranks.rbegin(), image_ranks.rend());

    cv::Mat disp_image_1 = cv::imread(image_ranks[1].second);
    cv::Mat disp_image_2 = cv::imread(image_ranks[2].second);
    cv::Mat disp_image_3 = cv::imread(image_ranks[3].second);
    std::cout<<image_ranks[1].second<<std::endl;
    std::cout<<image_ranks[2].second<<std::endl;
    std::cout<<image_ranks[3].second<<std::endl;
    cv::imshow("Best Match 1", disp_image_1);
    cv::imshow("Best Match 2", disp_image_2);
    cv::imshow("Best Match 3", disp_image_3);
    cv::imshow("Target Image", target_image);
    cv::waitKey(0);
    return 0;
}

int main(int argc, char *argv[]){

    if (argc<3){
        printf("usage: %s <directory path> <target image path>\n", argv[0]);
    }

    textureMatching(argv);
}