/*
Ronak Bhanushali
CS 5330
Spring 2024

This file applies filters to a given image
*/

#include "projects/project1/filters.hpp"
#include <cmath>

int greyscale(cv::Mat &src, cv::Mat &dst) {
    
    // Iterate over each pixel and apply the custom greyscale transformation
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            // Since images are in BGR format in OpenCV
            int blue = src.at<cv::Vec3b>(i, j)[0];
            int green = src.at<cv::Vec3b>(i, j)[1];
            int red = src.at<cv::Vec3b>(i, j)[2];

            // Calculate the greyscale value using a custom formula
            int greyValue = 255 - (red + green + blue) / 3;

            // Set the greyscale value to all channels in the destination matrix
            dst.at<cv::Vec3b>(i, j) = cv::Vec3b(greyValue, greyValue, greyValue);
        }
    }

    return 0;
}


int sepia(cv::Mat &src, cv::Mat &dst) {
    /*
    Sepia filter coefficients:
    0.272, 0.534, 0.131    // Blue coefficients
    0.349, 0.686, 0.168    // Green coefficients
    0.393, 0.769, 0.189    // Red coefficients
    */

   // Maximum pixel value for clamping
   float max_pixel_value = 255;

    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            
            // Extracting BGR values for the current pixel, since opencv have BGR and not RGB
            float blue = static_cast<float>(src.at<cv::Vec3b>(i, j)[0]);
            float green = static_cast<float>(src.at<cv::Vec3b>(i, j)[1]);
            float red = static_cast<float>(src.at<cv::Vec3b>(i, j)[2]);

            // Applying Sepia Filter
            float newBlue = 0.272 * blue + 0.534 * green + 0.131 * red;
            float newGreen = 0.349 * blue + 0.686 * green + 0.168 * red;
            float newRed = 0.393 * blue + 0.769 * green + 0.189 * red;

            // Clamp pixel values to ensure they are within the valid range [0, 255]
            newBlue = std::min(std::max(newBlue, 0.0f), max_pixel_value);
            newGreen = std::min(std::max(newGreen, 0.0f), max_pixel_value);
            newRed = std::min(std::max(newRed, 0.0f), max_pixel_value);

            // Setting the new pixel values in the destination image
            dst.at<cv::Vec3b>(i, j) = cv::Vec3b(static_cast<int>(newBlue), static_cast<int>(newGreen), static_cast<int>(newRed));
        }
    }
    return 0;
}
int sepia2(cv::Mat &src, cv::Mat &dst) {
    /*
    Sepia filter coefficients:
    0.272, 0.534, 0.131    // Blue coefficients
    0.349, 0.686, 0.168    // Green coefficients
    0.393, 0.769, 0.189    // Red coefficients
    */

    // Creating a 3x3 kernel matrix with Sepia coefficients
    cv::Mat kernel = cv::Mat::ones(cv::Size(3, 3), CV_32F);
    float max_pixel_value = 255;

    // Assigning Sepia filter coefficients to the kernel matrix
    kernel.at<float>(0, 0) = 0.272;
    kernel.at<float>(0, 1) = 0.534;
    kernel.at<float>(0, 2) = 0.131;
    kernel.at<float>(1, 0) = 0.349;
    kernel.at<float>(1, 1) = 0.686;
    kernel.at<float>(1, 2) = 0.168;
    kernel.at<float>(2, 0) = 0.393;
    kernel.at<float>(2, 1) = 0.769;
    kernel.at<float>(2, 2) = 0.189;

    // Iterating over each pixel in the source image
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            // Extracting BGR values for the current pixel
            cv::Mat pixel = cv::Mat::ones(cv::Size(1, 3), CV_32F);
            pixel.at<float>(0, 0) = static_cast<float>(src.at<cv::Vec3b>(i, j)[0]);
            pixel.at<float>(0, 1) = static_cast<float>(src.at<cv::Vec3b>(i, j)[1]);
            pixel.at<float>(0, 2) = static_cast<float>(src.at<cv::Vec3b>(i, j)[2]);

            // Applying Sepia Filter using the kernel matrix
            cv::Mat new_pixel = kernel * pixel;

            // Clamping pixel values to ensure they are within the valid range [0, 255]
            dst.at<cv::Vec3b>(i, j) = cv::Vec3b(
                static_cast<int>(std::min(new_pixel.at<float>(0, 0), max_pixel_value)),
                static_cast<int>(std::min(new_pixel.at<float>(0, 1), max_pixel_value)),
                static_cast<int>(std::min(new_pixel.at<float>(0, 2), max_pixel_value)));
        }
    }
    return 0;
}


int vignette(cv::Mat &src, cv::Mat &dst)
{
    int rows = src.rows;
    int cols = src.cols;
    float max_pixel_value = 255.0;
    float strength = 0.5;

    // Create a circular mask which can be used to multiply with each pixel
    cv::Mat mask = cv::Mat::zeros(src.size(), CV_32F);
    cv::Point center(cols / 2, rows / 2);
    float maxDistance = std::min(rows, cols) / 2.0;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float distance = cv::norm(cv::Point(j, i) - center);
            mask.at<float>(i, j) = 1.0 - strength * (distance / maxDistance);
        }
    }
    //Convert to float to apply vignetting
    src.convertTo(dst,CV_32F);
    std::vector<cv::Mat> channels;
    cv::split(dst, channels);

    for (cv::Mat& channel : channels) {
        channel = channel.mul(mask);
    }

    // Merge the channels to get the final output image
    cv::merge(channels, dst);
    dst.convertTo(dst, CV_8UC3);
    return 0;
}

int blur5x5_1_alt( cv::Mat &src, cv::Mat &dst ){
    int num_rows = src.rows;
    int num_cols = src.cols;
    int kernel[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}
    };
    cv::Mat temp_mat_1;
    src.convertTo(temp_mat_1,CV_32S);
    cv::Mat temp_mat_2 = temp_mat_1.clone();

    cv::Mat kernelMat = cv::Mat(5, 5, CV_32S);

    for (int i = 0; i < kernelMat.rows; ++i) {
        for (int j = 0; j < kernelMat.cols; ++j) {
            kernelMat.at<int>(i, j) = kernel[i][j];
        }
    }
    int denominator = cv::sum(kernelMat)[0];
    for (int i = 2; i < num_rows - 2; i++)
    {
        for (int j = 2; j < num_cols - 2; j++)
        {
            cv::Mat slicedMat = temp_mat_1(cv::Range(i - 2, i + 3), cv::Range(j - 2, j + 3));
            std::vector<cv::Mat> channels;
            cv::split(slicedMat, channels);
            int k = 0;
            for (cv::Mat &channel : channels)
            {
                int pixel_value = (cv::sum(kernelMat.mul(channel))[0])/denominator;
                temp_mat_2.at<cv::Vec3i>(i, j)[k] = pixel_value;
                k++;
            }
        }
    }
    
    dst = temp_mat_2.clone();
    dst.convertTo(dst, CV_8UC3);

    return 0;
}

int blur5x5_1( cv::Mat &src, cv::Mat &dst ){

	int num_rows = src.rows;
    int num_cols = src.cols;
    int kernel[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}
    };
    cv::Mat temp_mat_1;
    src.convertTo(temp_mat_1,CV_32S);
    cv::Mat temp_mat_2 = temp_mat_1.clone();

    int denominator = 100;
    for (int i = 2; i < num_rows - 2; i++)
    {
        for (int j = 2; j < num_cols - 2; j++)
        {
            cv::Vec3i pixel_value = cv::Vec3i (0,0,0);
            for (int k = 0; k < 5; k++)
            {
                for (int l = 0; l < 5; l++)
                {
                    pixel_value += kernel[k][l] * temp_mat_1.at<cv::Vec3i>(i+k-2,j+l-2);
                }    
            }
            pixel_value /= denominator;

            temp_mat_2.at<cv::Vec3i>(i,j) = pixel_value;
        }
    }
    temp_mat_2.convertTo(dst, CV_8UC3);
    return 0;
}

int blur5x5_2( cv::Mat &src, cv::Mat &dst ){

    int num_rows = src.rows;
    int num_cols = src.cols;

    cv::Mat temp_mat_1; //temporary matrix to clone src before processing

    src.convertTo(temp_mat_1,CV_32S);

    cv::Mat temp_mat_2 = temp_mat_1.clone(); //temporary matrix to store image after first convolution

    cv::Mat temp_mat_3 = temp_mat_2.clone(); //temporary matrix to store image after second convolution
    int denominator = 10;
	int kernel[5] = { 1,2,4,2,1 };

	for (int i = 2; i < num_rows - 2; i++)
	{
		for (int j = 2; j < num_cols - 2; j++)
        {
            cv::Vec3i pixel_value = cv::Vec3i(0, 0, 0);
            for (int k = 0; k < 5; k++)
            {
                pixel_value += kernel[k] * temp_mat_1.at<cv::Vec3i>(i+k-2,j);
            }
            temp_mat_2.at<cv::Vec3i>(i, j) = pixel_value / denominator;
        }
    }
	for (int i = 2; i < num_rows - 2; i++)
	{
		for (int j = 2; j < num_cols - 2; j++)
        {
            cv::Vec3i pixel_value = cv::Vec3i(0, 0, 0);
            for (int k = 0; k < 5; k++)
            {
                pixel_value += kernel[k] * temp_mat_2.at<cv::Vec3i>(i,j+k-2);
            }
            temp_mat_3.at<cv::Vec3i>(i, j) = pixel_value / denominator;
        }
    }
	temp_mat_3.convertTo(dst,CV_8UC3);
	return 0;
}

int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    int num_rows = src.rows;
    int num_cols = src.cols;

    cv::Mat temp_mat_1; //temporary matrix to clone src before processing

    src.convertTo(temp_mat_1,CV_32S);

    cv::Mat temp_mat_2 = temp_mat_1.clone(); //temporary matrix to store image after first convolution

    cv::Mat temp_mat_3 = temp_mat_2.clone(); //temporary matrix to store image after second convolution

    int kernel1[] = { -1, 0, 1 };
	int kernel2[] = { 1, 2, 1 };

//Vertical Kernel

	for (int i = 1; i < num_rows - 1; i++)
	{
		for (int j = 1; j < num_cols - 1; j++)
        {
            cv::Vec3i pixel_value = cv::Vec3i(0, 0, 0);
            for (int k = 0; k < 3; k++)
            {
                pixel_value += kernel1[k] * temp_mat_1.at<cv::Vec3i>(i,j+k-1);
            }
            temp_mat_2.at<cv::Vec3i>(i, j) = pixel_value;
        }
    }

//Horizontal Kernel

	for (int i = 1; i < num_rows - 1; i++)
	{
		for (int j = 1; j < num_cols - 1; j++)
        {
            cv::Vec3i pixel_value = cv::Vec3i(0, 0, 0);
            for (int k = 0; k < 3; k++)
            {
                pixel_value += kernel2[k] * temp_mat_2.at<cv::Vec3i>(i+k-1,j);
            }
            temp_mat_3.at<cv::Vec3i>(i, j) = pixel_value;
        }
    }
    cv::convertScaleAbs(temp_mat_3,dst,1); //Take absolute value of pixels since some might be negative. Note that scaling factor I used is 1 but can 0.25 can be used too. that would dim the edges a bi
	dst.convertTo(dst,CV_8UC3);
	return 0;
}

int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
    int num_rows = src.rows;
    int num_cols = src.cols;
    cv::Mat temp_mat_1;
    src.convertTo(temp_mat_1,CV_32S);
    cv::Mat temp_mat_2 = temp_mat_1.clone();
    cv::Mat temp_mat_3 = temp_mat_2.clone();
    int kernel1[] = { 1, 2, 1 };
	int kernel2[] = { -1, 0, 1 };

//Vertical Kernel

	for (int i = 1; i < num_rows - 1; i++)
	{
		for (int j = 1; j < num_cols - 1; j++)
        {
            cv::Vec3i pixel_value = cv::Vec3i(0, 0, 0);
            for (int k = 0; k < 3; k++)
            {
                pixel_value += kernel1[k] * temp_mat_1.at<cv::Vec3i>(i,j+k-1);
            }
            temp_mat_2.at<cv::Vec3i>(i, j) = pixel_value;
        }
    }

//Horizontal Kernel

	for (int i = 1; i < num_rows - 1; i++)
	{
		for (int j = 1; j < num_cols - 1; j++)
        {
            cv::Vec3i pixel_value = cv::Vec3i(0, 0, 0);
            for (int k = 0; k < 3; k++)
            {
                pixel_value += kernel2[k] * temp_mat_2.at<cv::Vec3i>(i+k-1,j);
            }
            temp_mat_3.at<cv::Vec3i>(i, j) = pixel_value;
        }
    }
    cv::convertScaleAbs(temp_mat_3,dst,1); //Take absolute value of pixels since some might be negative. Note that scaling factor I used is 1 but can 0.25 can be used too. that would dim the edges a bit
	dst.convertTo(dst,CV_8UC3);
	return 0;
}

int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst)
{
    int num_rows = sx.rows; //numver of rows
    int num_cols = sy.cols; //number of columns
    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_cols; j++)
        {
            cv::Vec3b x = sx.at<cv::Vec3b>(i, j); //get x gradient at pixel
            cv::Vec3b y = sy.at<cv::Vec3b>(i, j); //get y gradient at pixel

            for (int c = 0; c < 3; c++)
            {
                dst.at<cv::Vec3b>(i, j)[c] = sqrtf(pow(x[c], 2) + pow(y[c], 2)); //calculate gradient magnitude
            }
        }
    }
    return 0;
}

int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels )
{
    cv::Mat blurred_image;
    blur5x5_2(src, blurred_image);
    int b = 255 / levels;
    int xt,xf;
    for (int i = 0; i < blurred_image.rows; i++)
    {
        for (int j = 0; j < blurred_image.cols; j++)
        {
            cv::Vec3b bq = blurred_image.at<cv::Vec3b>(i, j);
            for (int c = 0; c < 3; c++) //iterate over channels
            {
                xt = bq[c] / b;
                xf = xt * b;
                dst.at<cv::Vec3b>(i, j)[c] = xf;
            }
        }
    }
    return 0;
}
int cartoon(cv::Mat &src, cv::Mat &dst, int levels, int threshold)
{
	cv::Mat mag(src.size(), CV_32S);
	cv::Mat sx(src.size(), CV_32S);
	cv::Mat sy(src.size(), CV_32S);
	sobelX3x3(src, sx);
	sobelY3x3(src, sy);
	magnitude(sx, sy, mag); //Get gradient Magnitude
	blurQuantize(src, dst, levels); //Blur quantize the image

	for (int i = 0; i < mag.rows; i++) 
	{
		for (int j = 0; j < mag.cols; j++) 
		{
			cv::Vec3b pixel = dst.at<cv::Vec3b>(i, j);
			cv::Vec3b magnitude = mag.at<cv::Vec3b>(i, j);

			for (int c = 0; c < 3; c++) {

				if (magnitude[c] > threshold) 
				{
					dst.at<cv::Vec3b>(i, j)[c] = 0;//changing all pixel values above threshold to black
				}
			}
		}
	}
    dst.convertTo(dst,CV_8UC3);
	return 0;
}

int applyBlurAndReplaceFace(cv::Mat& frame, cv::Mat& dst, const std::vector<cv::Rect>& faces) {

    cv::Mat temp = frame.clone(); //Clone input frame
    for (const auto& face : faces) {
        cv::Mat faceROI = frame(face).clone(); //Save face ROI

        blur5x5_2(frame, temp); //Blur image

        faceROI.copyTo(temp(face)); //Replace face location with unblurred face
    }
    dst = temp.clone();
    return 0;
}

int negativeImage(cv::Mat &src, cv::Mat &dst)
{
    cv::Mat temp(src.size(), CV_8UC3); //Clone Input
    for (int y = 0; y < src.rows; ++y)
    {
        for (int x = 0; x < src.cols; ++x)
        {
            // Get the pixel at (x, y)
            cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);

            // Invert the color values for each channel (BGR)
            pixel[0] = 255 - pixel[0]; // Blue
            pixel[1] = 255 - pixel[1]; // Green
            pixel[2] = 255 - pixel[2]; // Red

            // Set the modified pixel to the output image
            temp.at<cv::Vec3b>(y, x) = pixel;
        }
    }
    dst = temp.clone();
    return 0;
}