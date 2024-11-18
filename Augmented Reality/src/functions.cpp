/*

functions.cpp
Project 4

Created by Ronak Bhanushali and Rucha Pendharkar on 2/8/24

This file contains functions used for multiple tasks.


*/

#include "functions.hpp"

int generateArUcoMarkersGrid(int num_markers, int spacing = 500)
{
    //Specify Dictionary for generating aruco markers
    cv::aruco::Dictionary aruco_dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    //MarkerSize and spacing parameters
    int marker_size = 200;
    int border_size = 1;
    int cols = std::ceil(std::sqrt(num_markers));
    int rows = std::ceil((float)num_markers / cols);
    int grid_width = cols * marker_size + (cols - 1) * spacing;
    int grid_height = rows * marker_size + (rows - 1) * spacing;
    int top_margin = spacing/4;
    int bottom_margin = spacing/4;
    int left_margin = spacing/4;
    int right_margin = spacing/4;

    //Create a grid
    cv::Mat grid(grid_height + top_margin + bottom_margin, grid_width + left_margin + right_margin, CV_8UC1, cv::Scalar(255));

    //Populate the grid
    for (int i = 0; i < num_markers; ++i)
    {
        int row = i / cols;
        int col = i % cols;
        cv::Mat marker_image;
        cv::aruco::generateImageMarker(aruco_dict,i, marker_size, marker_image, border_size);
        int x_offset = col * (marker_size + spacing) + left_margin;
        int y_offset = row * (marker_size + spacing) + top_margin;
        cv::Rect roi(x_offset, y_offset, marker_size, marker_size);
        marker_image.copyTo(grid(roi));
    }

    //Show image and save
    cv::imshow("ArUco Markers Grid", grid);
    cv::waitKey(0);
    cv::imwrite("aruco.jpg", grid);
    cv::destroyAllWindows();
    exit(0);
    return 0;
}

int generateArUcoMarkers(int num_markers, int marker_size = 200, int spacing = 50)
{
    // Define parameters for the markers
    cv::aruco::Dictionary aruco_dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    int border_size = 1;

    // Create the grid image
    int grid_width = marker_size * (num_markers + 1) + spacing * num_markers;
    int grid_height = marker_size * (num_markers + 1) + spacing * num_markers;
    cv::Mat grid(grid_height, grid_width, CV_8UC1, cv::Scalar(255));

    // Add a marker in the center
    int center_x = grid_width / 2 - marker_size / 2;
    int center_y = grid_height / 2 - marker_size / 2;
    cv::Mat center_marker;
    cv::aruco::generateImageMarker(aruco_dict, 0, marker_size, center_marker, border_size);
    cv::Rect center_roi(center_x, center_y, marker_size, marker_size);
    center_marker.copyTo(grid(center_roi));

    // Arrange other markers around the center in a circular pattern
    double radius = marker_size * (num_markers / 2 + 1.5);
    for (int i = 1; i < num_markers; ++i)
    {
        double angle = 2 * CV_PI * i / (num_markers - 1);
        int x_offset = center_x + radius * std::cos(angle) - marker_size / 2;
        int y_offset = center_y + radius * std::sin(angle) - marker_size / 2;
        cv::Rect roi(x_offset, y_offset, marker_size, marker_size);
        cv::Mat marker_image;
        cv::aruco::generateImageMarker(aruco_dict, i, marker_size, marker_image, border_size);
        marker_image.copyTo(grid(roi));
    }

    // Display the grid with markers
    cv::imshow("ArUco Markers Grid", grid);
    cv::waitKey(0);
    cv::imwrite("aruco.jpg", grid);

    return 0;
}

std::pair<std::vector<int> ,std::vector<std::vector<cv::Point2f>>> getTarget(cv::Mat &frame, cv::Mat &detected_frame)
{

    //Convert to grayscale
    cv::Mat gray;
    detected_frame = frame.clone();
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    //Define dictionary and detector params
    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    cv::aruco::Dictionary aruco_dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    std::vector<std::vector<cv::Point2f>> corners, rejected_candidates;
    std::vector<int> ids;

    //Make detector object
    cv::aruco::ArucoDetector detector(aruco_dict, detectorParams);

    //Detect markers from frame
    detector.detectMarkers(gray, corners, ids, rejected_candidates);

    //Draw markers
    cv::aruco::drawDetectedMarkers(detected_frame, corners, ids);

    //return ids and corresponding corner points
    return std::make_pair(ids,corners);
}

int loadParameters(const std::string &file_path, cv::Mat &camera_matrix, cv::Mat &dist_coeffs)
{
    // Load the XML or YAML file containing camera parameters
    cv::FileStorage fs(file_path, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "Failed to open file!" << std::endl;
        return -1;
    }

    fs["cameraMatrix"] >> camera_matrix;
    fs["distCoeffs"] >> dist_coeffs;
    fs.release();

    return 0;
}

int undistortImage(const cv::Mat &input_image, const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, cv::Mat &undistorted_image)
{
    // Get the size of the input image
    cv::Size image_size = input_image.size();

    // Compute the rectification map
    cv::Mat map1, map2;
    cv::Mat new_camera_matrix = cv::getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, image_size, 1, image_size, 0);
    cv::initUndistortRectifyMap(camera_matrix, dist_coeffs, cv::Mat(), new_camera_matrix, image_size, CV_16SC2, map1, map2);

    // Undistort the image
    cv::remap(input_image, undistorted_image, map1, map2, cv::INTER_LINEAR);
    return 0;
}