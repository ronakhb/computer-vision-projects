/*

task5.cpp
Project 4

Created by Ronak Bhanushali and Rucha Pendharkar on 2/8/24

This code is used for task5


*/

#include "functions.hpp"

int main()
{
    // Open the video capture device
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Unable to open video device" << std::endl;
        return -1;
    }
    // Create a named window for displaying the video feed
    cv::namedWindow("Video Feed", cv::WINDOW_AUTOSIZE);
    cv::Mat frame, detected_frame;
    cv::Mat camera_matrix, dist_coeffs, undistorted_image;
    cv::Mat rvec, tvec;

    std::vector<std::vector<cv::Point2f>> corners;

    // Define the object points corresponding to the ArUco marker
    std::vector<cv::Point3f> object_points;
    object_points.push_back(cv::Point3f(0, 0, 0));
    object_points.push_back(cv::Point3f(0.026, 0, 0));
    object_points.push_back(cv::Point3f(0.026, 0.026, 0));
    object_points.push_back(cv::Point3f(0, 0.026, 0));

    // Load camera parameters from file
    std::string file_path = "data/calibration.xml";
    loadParameters(file_path, camera_matrix, dist_coeffs);
    std::vector<cv::Point2f> origin_target;

    while (true)
    {
        // Capture frame from the video
        cap >> frame;

        // Check if the frame is empty
        if (frame.empty())
            break;


        // Undistort the frame using camera calibration parameters
        undistortImage(frame, camera_matrix, dist_coeffs, undistorted_image);
        auto target_data = getTarget(undistorted_image, detected_frame);
        auto ids = target_data.first;
        auto corners = target_data.second;

        // If no corners are detected, exit the loop
        if (corners.size() == 0)
        {
            cv::imshow("Video Feed", detected_frame);
            int key = cv::waitKey(17);
            // Exit the loop if 'q' is pressed
            if (key == 'q')
            {
                cv::destroyAllWindows();
                return 0;
            }
            continue;
        }

        // Get the corners of the first detected marker
        origin_target = corners[0];

        for (int i = 0; i < ids.size(); i++)
        {
            if (ids[i] == 0)
            {
                origin_target = corners[i];
            }

            std::vector<cv::Point2f> projection_points;

            // Find camera position using solvePnP
            cv::solvePnP(object_points, corners[i], camera_matrix, dist_coeffs, rvec, tvec);
            cv::projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs, projection_points);

            // Draw projected points on the detected frame
            for (int i = 0; i < projection_points.size(); i++)
            {
                cv::Point2d point = projection_points[i];
                cv::circle(detected_frame, point, 5, cv::Scalar(255, 0, 255), -1);
            }
        }
        // Draw origin point and axis lines on the detected frame
        cv::circle(detected_frame, origin_target[0], 5, cv::Scalar(0, 0, 255), -1);
        cv::arrowedLine(detected_frame, origin_target[0], origin_target[1], cv::Scalar(0, 0, 255), 2);
        cv::arrowedLine(detected_frame, origin_target[0], origin_target[3], cv::Scalar(0, 255, 0), 2);

        // Display the detected frame
        cv::imshow("Video Feed", detected_frame);
        int key = cv::waitKey(17);
        // Exit the loop if 'q' is pressed
        if (key == 'q')
        {
            cv::destroyAllWindows();
            return 0;
        }
    }
    return 0;
}