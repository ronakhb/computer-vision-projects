/*

calibration.cpp
Project 4

Created by Ronak Bhanushali and Rucha Pendharkar on 2/8/24

This code is used for camera calibration.


*/

#include <opencv2/opencv.hpp>
#include <vector>

int main(int argc, char *argv[])
{
    // open the video device
    cv::VideoCapture *vidCap;
    vidCap = new cv::VideoCapture(0);

    // Error Checking
    if (!vidCap->isOpened())
    {
        std::cerr << "Error: Unable to open video device!" << std::endl;
        return (-1);
    }

    // Get size of the image
    cv::Size refS((int)vidCap->get(cv::CAP_PROP_FRAME_WIDTH),
                   (int)vidCap->get(cv::CAP_PROP_FRAME_HEIGHT));

    // The number of corners
    int cornersPerRow = 9;
    int cornersPerCol = 6;
    cv::Size checkerboard_pattern = cv::Size(cornersPerRow, cornersPerCol);

    std::vector<cv::Vec3f> pointSet;
    for (int i = 0; i < cornersPerCol; i++)
    {
        for (int j = 0; j < cornersPerRow; j++)
        {
            pointSet.push_back(cv::Vec3f(j, -i, 0));
        }
    }

    // Create vectors to store corners and points during calibration
    std::vector<std::vector<cv::Point2f>> cornerList;
    std::vector<std::vector<cv::Vec3f>> pointList;
    cv::Mat frame;
    while (true)
    {
        *vidCap >> frame;

        if (frame.empty())
        {
            std::cerr << "Error: Frame is empty" << std::endl;
            break;
        }

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // find the chessboard corners

        std::vector<cv::Point2f> cornerSet;
        cv::TermCriteria termCrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
        bool found = cv::findChessboardCorners(gray, checkerboard_pattern, cornerSet);

        if (found)
        {
            // refine the corner locations
            cv::cornerSubPix(gray, cornerSet, cv::Size(11, 11), cv::Size(-1, -1), termCrit);

            // draw the corners on the frame
            cv::drawChessboardCorners(frame, checkerboard_pattern, cornerSet, found);
        }

        // display the frame
        cv::imshow("Calibration Image with chessboard corners ", frame);

        int key = cv::waitKey(1);

        // if key is 's', store the corner locations and 3D points, and save the image
        if (key == 's')
        {
            if (!found)
            {
                std::cerr << "Corners not found!" << std::endl;
                continue;
            }

            //Save corner locations and 3D points
            cornerList.push_back(cornerSet);
            pointList.push_back(pointSet);
            cv::imwrite("data/CalibrationImage.png", frame); // Update file path as needed


            //If number of images are greater than 5, perform camera calibration.
            if (cornerList.size() >= 5)
            {
                std::cout << "5 Images captured successfully. Starting Calibration!" << std::endl;
                cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64FC1);
                cameraMatrix.at<double>(0, 2) = refS.width / 2;
                cameraMatrix.at<double>(1, 2) = refS.height / 2;
                cv::Mat distCoeffs = cv::Mat::zeros(1, 14, CV_64FC1);

                std::cout << "Camera Matrix before calibration" << cameraMatrix << std::endl;
                std::cout << "Distortion Co-efficients before calibration" << distCoeffs << std::endl;
            
                std::vector<cv::Mat> rotationVecs, translationVecs; 
                double reprojectionError = cv::calibrateCamera(pointList, cornerList, refS, cameraMatrix, distCoeffs, rotationVecs, translationVecs, cv::CALIB_FIX_ASPECT_RATIO | cv::CALIB_RATIONAL_MODEL, termCrit);

                std::cout << "Camera Matrix after calibration: " << cameraMatrix << '\n' << std::endl;
                std::cout << "Distortion Co-efficients after calibration: " << distCoeffs << '\n' << std::endl;
                std::cout << "Reprojection Error: " << reprojectionError << '\n' << std::endl;

                // Save camera matrix and distortion coefficients to an XML file
                cv::FileStorage fs("data/calibration.xml", cv::FileStorage::WRITE); //Update file path as needed
                fs << "cameraMatrix" << cameraMatrix;
                fs << "distCoeffs" << distCoeffs;
                fs.release();

                std::cout << "Calibration data saved to calibration.xml"<< '\n' << std::endl;

                break;
            }
            else
            {
                std::cout <<"5 Images are required for calibration!"<< '\n'<< std::endl;
            }
        }
        //If user presses q, exit the application
        else if (key == 'q'){ 
            break;
        }
    }

    delete vidCap;

    return (0);
}
