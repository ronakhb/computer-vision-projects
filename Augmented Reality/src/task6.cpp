/*

task6.cpp
Project 4

Created by Ronak Bhanushali and Rucha Pendharkar on 2/8/24

This code is used for task6


*/

#include "functions.hpp"

// Function to draw a tent on the detected frame
int drawTent(cv::Mat &src, std::vector<cv::Point2f> &projected_points)
{
    // Draw lines representing the tent structure
    cv::line(src, projected_points[1], projected_points[0], cv::Scalar(255, 0, 255), 2);
    cv::line(src, projected_points[2], projected_points[1], cv::Scalar(255, 0, 255), 2);
    cv::line(src, projected_points[3], projected_points[2], cv::Scalar(255, 0, 255), 2);
    cv::line(src, projected_points[0], projected_points[3], cv::Scalar(255, 0, 255), 2);
    cv::line(src, projected_points[2], projected_points[4], cv::Scalar(255, 0, 255), 2);
    cv::line(src, projected_points[3], projected_points[4], cv::Scalar(255, 0, 255), 2);
    cv::line(src, projected_points[4], projected_points[5], cv::Scalar(255, 0, 255), 2);
    cv::line(src, projected_points[0], projected_points[5], cv::Scalar(255, 0, 255), 2);
    cv::line(src, projected_points[1], projected_points[5], cv::Scalar(255, 0, 255), 2);

    return 0;
}

// Function to draw a human figure on the detected frame
int drawHuman(cv::Mat &src, std::vector<cv::Point2f> &projected_points)
{
    // Draw lines representing the human figure
    cv::line(src, projected_points[0], projected_points[2], cv::Scalar(255, 0, 255), 2);
    cv::line(src, projected_points[1], projected_points[2], cv::Scalar(255, 0, 255), 2);
    cv::line(src, projected_points[2], projected_points[3], cv::Scalar(255, 0, 255), 2);
    cv::line(src, projected_points[4], projected_points[3], cv::Scalar(255, 0, 255), 2);
    cv::line(src, projected_points[3], projected_points[5], cv::Scalar(255, 0, 255), 2);
    cv::line(src, projected_points[3], projected_points[6], cv::Scalar(255, 0, 255), 2);
    cv::line(src, projected_points[8], projected_points[projected_points.size() - 1], cv::Scalar(255, 0, 255), 2);
    for (int i = 8; i < projected_points.size() - 1; i++)
    {
        cv::line(src, projected_points[i], projected_points[i + 1], cv::Scalar(255, 0, 255), 2);
    }

    return 0;
}

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
    object_points.push_back(cv::Point3f(0.0185, 0, 0));
    object_points.push_back(cv::Point3f(0.0185, 0.0185, 0));
    object_points.push_back(cv::Point3f(0, 0.0185, 0));

    // Define points for drawing a cube
    std::vector<cv::Point3f> cube;
    cube.push_back(cv::Point3f(0, 0, -0.03));
    cube.push_back(cv::Point3f(0.026, 0, -0.03));
    cube.push_back(cv::Point3f(0.026, 0.026, -0.03));
    cube.push_back(cv::Point3f(0, 0.026, -0.03));

    // Define points for drawing a tent
    std::vector<cv::Point3f> tent;
    tent.push_back(cv::Point3f(0, 0, 0));
    tent.push_back(cv::Point3f(0.05, 0, 0));
    tent.push_back(cv::Point3f(0.05, 0.05, 0));
    tent.push_back(cv::Point3f(0, 0.05, 0));
    tent.push_back(cv::Point3f(0.05, 0.05, -0.025));
    tent.push_back(cv::Point3f(0.05, 0, -0.05));

    // Define points for drawing a human figure
    std::vector<cv::Point3f> human;
    human.push_back(cv::Point3f(0, 0.025, 0));
    human.push_back(cv::Point3f(0.05, 0.025, 0));
    human.push_back(cv::Point3f(0.025, 0.025, -0.025));
    human.push_back(cv::Point3f(0.025, 0.025, -0.05));
    human.push_back(cv::Point3f(0, 0.025, -0.09));
    human.push_back(cv::Point3f(0.05, 0.025, -0.09));
    human.push_back(cv::Point3f(0.025, 0.025, -0.07));
    human.push_back(cv::Point3f(0.025, 0.025, -0.08));
    
    // Generate points around the human figure to represent motion
    for (int i = 0; i < 20; ++i)
    {
        double angle = 2 * CV_PI * i / 20;
        double x = 0.025 + 0.01 * std::cos(angle);
        double z = 0.08 + 0.01 * std::sin(angle);
        human.push_back(cv::Point3d(x, 0.025, -z));
    }

    std::string file_path = "data/calibration.xml";
    // Load camera parameters from file
    loadParameters(file_path, camera_matrix, dist_coeffs);
    std::vector<cv::Point2f> origin_target;
    bool is_first_iteration = true;

    while (true)
    {
        // Capture frame from the video
        cap >> frame;

        // Check if the frame is empty
        if (frame.empty())
            continue;

        undistortImage(frame, camera_matrix, dist_coeffs, undistorted_image);
        auto target_data = getTarget(undistorted_image, detected_frame);
        auto ids = target_data.first;
        auto corners = target_data.second;

        // If no corners are detected, continue to the next iteration
        if (corners.size() == 0)
        {
            cv::imshow("Video Feed", detected_frame);
            int key = cv::waitKey(17);
            if (key == 'q')
            {
                cv::destroyAllWindows();
                return 0;
            }
            continue;
        }

        origin_target = corners[0];

        for (int i = 0; i < ids.size(); i++)
        {
            std::vector<cv::Point2f> projection_points;
            bool pnp_flag = true;

            if (ids[i] == 0)
            {
                origin_target = corners[i];
                // Solve PnP for the tent object
                cv::solvePnP(object_points, corners[i], camera_matrix, dist_coeffs, rvec, tvec, pnp_flag);
                cv::projectPoints(tent, rvec, tvec, camera_matrix, dist_coeffs, projection_points);
                // Draw the tent on the detected frame
                drawTent(detected_frame, projection_points);
                continue;
            }
            // Skip PnP calculation for the first iteration to avoid flickering
            if (is_first_iteration)
            {
                pnp_flag = false;
                is_first_iteration = false;
            }
            // Solve PnP for the human figure object
            cv::solvePnP(object_points, corners[i], camera_matrix, dist_coeffs, rvec, tvec, pnp_flag);
            cv::projectPoints(human, rvec, tvec, camera_matrix, dist_coeffs, projection_points);
            // Draw the human figure on the detected frame
            drawHuman(detected_frame, projection_points);
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