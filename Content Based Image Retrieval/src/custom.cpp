/*
Prem Sukhadwala and Ronak Bhanushali
January 2024
Code for Content Based Image Retrieval using color features and DNN features
*/

#include <iostream> //standard input output
#include <opencv2/opencv.hpp>
#include "functions.hpp"   
#include "texture.hpp"

int main (int argc, char* argv[])
{
    //checks for sufficient arguments
    if( argc < 3) {
    printf("usage: %s <directory path>\n", argv[0]);
    exit(-1);
    }    

    customHistMatching(argv);

    return(0);
}
