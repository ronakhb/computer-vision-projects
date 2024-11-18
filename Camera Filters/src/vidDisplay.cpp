/*
Ronak Bhanushali
CS 5330
Spring 2024

This file displays an image with specified path
*/

#include "projects/project1/vidDisplay.hpp"
#include "projects/project1/filters.hpp"
#include "projects/project1/faceDetect.h"

int g_video_mode = 0; //Global Variable to Control which filter to apply
int g_vignette = 0; //Global Variable to Control whether to apply vignetting
int g_video_recorder = 0; //Global Variable to control whether to save video
int g_frame_number = 0; //Global Variable to keep count of frame numbers for screenshots
std::vector<cv::Mat> save_frames; //Frames to be saved as video
std::string g_save_location = "output_files/";

int displayVideo()
{
    cv::VideoCapture *capdev;
    // open the video device
    capdev = new cv::VideoCapture(0,cv::CAP_V4L2);
    capdev->set(cv::CAP_PROP_BUFFERSIZE, 2);

    if (!capdev->isOpened())
    {
        printf("Unable to open video device\n");
        return (-1);
    }

    // get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("Video", 1); // identifies a window
    cv::Mat frame;

    while (true)
    {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty())
        {
            printf("frame is empty\n");
            break;
        }

        // Apply selected video mode
        switch (g_video_mode) {
            case GRAYSCALE:
                cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
                break;
            case GREYSCALE_CUSTOM:
                greyscale(frame, frame);
                break;
            case SEPIA:
                sepia(frame, frame);
                break;
            case BLUR_5X5:
                blur5x5_2(frame, frame);
                break;
            case SOBEL_X_3X3:
                sobelX3x3(frame, frame);
                break;
            case SOBEL_Y_3X3:
                sobelY3x3(frame, frame);
                break;
            case MAGNITUDE_SOBEL: {
                cv::Mat sx, sy;
                sobelX3x3(frame, sx);
                sobelY3x3(frame, sy);
                magnitude(sx, sy, frame);
                break;
            }
            case BLUR_QUANTIZE:
                blurQuantize(frame, frame, 5);
                break;
            case FACE_DETECTION: {
                cv::Mat grey_image;
                std::vector<cv::Rect> faces;
                cv::cvtColor(frame, grey_image, cv::COLOR_BGR2GRAY);
                detectFaces(grey_image, faces);
                drawBoxes(frame, faces);
                break;
            }
            case CARTOONIZE:
                cartoon(frame, frame, 5, 50);
                break;
            case FACE_BLUR: {
                cv::Mat grey_image;
                std::vector<cv::Rect> faces;
                cv::cvtColor(frame, grey_image, cv::COLOR_BGR2GRAY);
                detectFaces(grey_image, faces);
                applyBlurAndReplaceFace(frame, frame, faces);
                drawBoxes(frame, faces);
                break;
            }
            case NEGATIVE_IMAGE:
                negativeImage(frame, frame);
                break;

            default:
                break;
        }

        // Apply vignette if enabled
        if (g_vignette == 1) {
            vignette(frame, frame);
        }

        // Save frames if video recording is enabled
        if (g_video_recorder == 1) {
            save_frames.push_back(frame.clone());
        }

        cv::imshow("Video", frame);
        // see if there is a waiting keystroke


        char key = cv::waitKey(17);
        if (key == 'q')
        {
            break;
        }
        else if (key == 's')
        {
            std::string file_name = g_save_location + std::to_string(g_frame_number) + ".jpg";
            cv::imwrite(file_name,frame);
        }
        else if (key == 'g')
        {
            if (g_video_mode != 1)
            {
                g_video_mode = 1;
            }
            else
            {
                g_video_mode = 0;
            }
        }
        else if (key == 'h')
        {
            if (g_video_mode != 2)
            {
                g_video_mode = 2;
            }
            else
            {
                g_video_mode = 0;
            }
        }
        else if (key == 'p')
        {
            if (g_video_mode != 3)
            {
                g_video_mode = 3;
            }
            else
            {
                g_video_mode = 0;
            }
        }
        else if (key == 'b')
        {
            if (g_video_mode != 4)
            {
                g_video_mode = 4;
            }
            else
            {
                g_video_mode = 0;
            }
        }
        else if (key == 'x')
        {
            if (g_video_mode != 5)
            {
                g_video_mode = 5;
            }
            else
            {
                g_video_mode = 0;
            }
        }
        else if (key == 'y')
        {
            if (g_video_mode != 6)
            {
                g_video_mode = 6;
            }
            else
            {
                g_video_mode = 0;
            }
        }
        else if (key == 'm')
        {
            if (g_video_mode != 7)
            {
                g_video_mode = 7;
            }
            else
            {
                g_video_mode = 0;
            }
        }
        else if (key == 'i')
        {
            if (g_video_mode != 8)
            {
                g_video_mode =8;
            }
            else
            {
                g_video_mode = 0;
            }
        }
        else if (key == 'f')
        {
            if (g_video_mode != 9)
            {
                g_video_mode = 9;
            }
            else
            {
                g_video_mode = 0;
            }
        }
        else if (key == 'c')
        {
            if (g_video_mode != 10)
            {
                g_video_mode = 10;
            }
            else
            {
                g_video_mode = 0;
            }
        }
        else if (key == 'a')
        {
            if (g_video_mode != 11)
            {
                g_video_mode = 11;
            }
            else
            {
                g_video_mode = 0;
            }
        }
        else if (key == 'n')
        {
            if (g_video_mode != 12)
            {
                g_video_mode = 12;
            }
            else
            {
                g_video_mode = 0;
            }
        }
        else if (key == 'v')
        {
            if (g_vignette != 1)
            {
                g_vignette = 1;
            }
            else
            {
                g_vignette = 0;
            }
        }
        else if (key == 'r')
        {
            if (g_video_recorder != 1)
            {
                g_video_recorder = 1;
                std::cout<<"Saving Video"<<std::endl;
            }
            else
            {
                g_video_recorder = 0;
                cv::VideoWriter videoWriter(g_save_location + "output_video.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, cv::Size((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH), (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT)));
                if (!videoWriter.isOpened()) {
                    std::cerr << "Error: Could not open the video writer." << std::endl;
                }
                for (const auto &save_frame : save_frames)
                {
                    // Write the frame to the video
                    videoWriter.write(save_frame);
                }
                videoWriter.release();
                std::cout<<"Saved Video"<<std::endl;
            }
        }
        g_frame_number++;
    }

    delete capdev;
    return (0);
}