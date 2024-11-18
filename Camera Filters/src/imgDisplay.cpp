/*
Ronak Bhanushali
CS 5330
Spring 2024

This file displays an image with specified path
*/

#include "projects/project1/imgDisplay.hpp"

uint g_fullscreen=0;

void toggleFullScreen(cv::Mat img)
{
    if (g_fullscreen == 0)
    {
        cv::destroyAllWindows();
        cv::namedWindow("Task 1", cv::WINDOW_NORMAL);
        cv::setWindowProperty("Task 1", cv::WND_PROP_FULLSCREEN,
                              cv::WINDOW_FULLSCREEN);
        cv::Size screen_resolution = cv::getWindowImageRect("Task 1").size();
        // Resize the image to match the screen resolution
        //Tried to resize image but I always get -1,-1 as width and height
        // cv::resize(img, img, screen_resolution);
        cv::imshow("Task 1", img);
        g_fullscreen = 1;
    }
    else
    {
        cv::destroyAllWindows();
        cv::imshow("Task 1", img);
        g_fullscreen = 0;
    }
}

int displayImage(std::string path_to_img)
{
    cv::Mat img = cv::imread(path_to_img);
    cv::imshow("Task 1", img);
    while (true)
    {
        int keypress = cv::waitKey(0);
        if (keypress == 'q')
        {
            cv::destroyAllWindows();
            break;
        }
        else if (keypress == 'f')
        {
            toggleFullScreen(img);
        }
    }
    return 0;
}