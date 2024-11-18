/*

task7.cpp
Project 4

Created by Ronak Bhanushali and Rucha Pendharkar on 2/8/24

This code is used for Task 7.


*/
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>

int main(int argc, char *argv[])
{
    // Open the video device
    cv::VideoCapture vidCap(0);

    // Error Checking
    if (!vidCap.isOpened())
    {
        std::cerr << "Error: Unable to open video device!" << std::endl;
        return -1;
    }

    // Create a SURF object
    int minHessian = 500;
    cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(minHessian);

    // Create a SIFT object
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(minHessian);
    // Create a Harris corner detector object
    cv::Ptr<cv::FeatureDetector> harris = cv::GFTTDetector::create(minHessian);

    // Vector to store keypoints
    std::vector<cv::KeyPoint> surfKeypoints;
    std::vector<cv::KeyPoint> siftKeypoints;
    std::vector<cv::KeyPoint> harrisKeypoints;

    while (true)
    {
        cv::Mat frame;
        vidCap.read(frame);

        if (frame.empty())
        {
            std::cerr << "Error: Unable to read frame from video!" << std::endl;
            break;
        }

        // Detect SURF keypoints
        surf->detect(frame, surfKeypoints);

        // Draw SURF keypoints on the frame
        cv::Mat frameWithSURFKeypoints;
        cv::drawKeypoints(frame, surfKeypoints, frameWithSURFKeypoints);

        // Detect SIFT keypoints
        sift->detect(frame, siftKeypoints);

        // Draw SIFT keypoints on the frame
        cv::Mat frameWithSIFTKeypoints;
        cv::drawKeypoints(frame, siftKeypoints, frameWithSIFTKeypoints);

        // Detect Harris keypoints
        harris->detect(frame, harrisKeypoints);

        // Draw Harris keypoints on the frame
        cv::Mat frameWithHarrisKeypoints;
        cv::drawKeypoints(frame, harrisKeypoints, frameWithHarrisKeypoints, cv::Scalar(0, 255, 0));

        // Display the frames with keypoints
        cv::imshow("SURF Keypoints", frameWithSURFKeypoints);
        cv::imshow("SIFT Keypoints", frameWithSIFTKeypoints);
        cv::imshow("Harris Keypoints", frameWithHarrisKeypoints);

        // Check for 'q' key to quit
        if (cv::waitKey(10) == 'q')
            break;
    }

    // Release the video capture object
    vidCap.release();

    // Close all OpenCV windows
    cv::destroyAllWindows();

    return 0;
}

