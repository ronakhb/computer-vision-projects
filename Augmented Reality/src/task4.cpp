/*

task4.cpp
Project 4

Created by Ronak Bhanushali and Rucha Pendharkar on 2/8/24

This code is used for task4


*/

#include "functions.hpp"

int main()
{
   cv::VideoCapture vidCap(0);

    if (!vidCap.isOpened())
    {
        std::cerr << "Error: Unable to open video device!" << std::endl;
        return -1;
    }

    //Create window
    cv::namedWindow("Video Feed", cv::WINDOW_AUTOSIZE);

    //Define variables
    cv::Mat frame, detected_frame;
    cv::Mat camera_matrix, dist_coeffs, undistorted_image;
    cv::Mat rvec, tvec;

    //Variable to hold aruco corner points
    std::vector<std::vector<cv::Point2f>> corners;

    //Real dimension of aruco marker in meters
    std::vector<cv::Point3f> object_points;
    object_points.push_back(cv::Point3f(0, 0, 0));
    object_points.push_back(cv::Point3f(0.026, 0, 0));
    object_points.push_back(cv::Point3f(0.026, 0.025, 0));
    object_points.push_back(cv::Point3f(0, 0.026, 0));

    //Load calibration parameters into calilbration matrix
    std::string file_path = "data/calibration.xml";
    loadParameters(file_path, camera_matrix, dist_coeffs);

    //Variable to hold corner points of aruco with id0
    std::vector<cv::Point2f> origin_target;

    while (true)
    {
        vidCap.read(frame);

            if (frame.empty())
                continue;

        //Undistort Image
        undistortImage(frame, camera_matrix, dist_coeffs, undistorted_image);
        auto target_data = getTarget(undistorted_image, detected_frame);
        auto ids = target_data.first;
        auto corners = target_data.second;

        //Generate aruco campsite
        // generateArUcoMarkers(7, 200,200);

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
            if (ids[i] == 0)
            {
                origin_target = corners[i];
            }
        }

        //Draw origin
        cv::circle(detected_frame, origin_target[0], 5, cv::Scalar(0, 0, 255), -1);
        cv::arrowedLine(detected_frame, origin_target[0], origin_target[1], cv::Scalar(0, 0, 255), 2);
        cv::arrowedLine(detected_frame, origin_target[0], origin_target[3], cv::Scalar(0, 255, 0), 2);

        //Find camera position
        cv::solvePnP(object_points, corners[0], camera_matrix, dist_coeffs, rvec, tvec);
        std::cout << "Translation" << tvec << std::endl;
        std::cout << "Rotation" << rvec << std::endl;

        cv::imshow("Video Feed", detected_frame);
        int key = cv::waitKey(17);
        if (key == 'q')
        {
            cv::destroyAllWindows();
            return 0;
        }
    }
    vidCap.release();

    return 0;
}