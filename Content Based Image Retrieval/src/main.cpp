/*
Prem Sukhadwala and Ronak Bhanushali
January 2024
Code for Content Based Image Retrieval using color features and DNN features
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

    //Task 1
    //command line arguments order --> image directory path, csv file path, target image PATH
    writeFeaturesToCSV(argv);
    compareFeatures(argv);

    //Task 5
    //command line arguments order --> image directory path, csv file path, target image NAME
    // compareEmbeddedVectors(argv);


    return(0);
}
