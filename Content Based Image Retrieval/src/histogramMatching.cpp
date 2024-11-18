/*
Prem Sukhadwala and Ronak Bhanushali
January 2024
Code for Content Based Image Retrieval using histogram features
*/

#include <iostream> //standard input output
#include <opencv2/opencv.hpp>
#include "../include/functions.hpp"   

using namespace cv;

int main(int argc, char *argv[]){

    //checks for sufficient arguments
    if( argc < 4) {
    printf("usage: %s <directory path>\n", argv[0]);
    exit(-1);
    }      

    
    if (strcmp(argv[3], "single") == 0)
    {
        histogramMatching(argv);
    }
    else if (strcmp(argv[3], "multi") == 0)
    {
        multiHistogramMatching(argv);
    }

    return (0);
}
