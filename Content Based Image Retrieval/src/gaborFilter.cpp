#include <gaborFilter.hpp>

int applyGaborFilter(cv::Mat& input_image, cv::Mat& filtered_image, int width, int height, double sigma, double theta, double lambda, double gamma, double psi) {
    cv::Mat kernel(width, height, CV_32F);

    double sigma_x = sigma;
    double sigma_y = sigma / gamma;

    double half_width = (width - 1) / 2;
    double half_height = (height - 1) / 2;

    for (int y = -half_height; y <= half_height; ++y) {
        for (int x = -half_width; x <= half_width; ++x) {

            double x_theta = x * std::cos(theta) + y * std::sin(theta);
            double y_theta = -x * std::sin(theta) + y * std::cos(theta);


            double gauss = std::exp(-(x_theta * x_theta + gamma * gamma * y_theta * y_theta) / (2 * sigma_x * sigma_y));

            double sinusoid = std::cos(2 * M_PI * x_theta / lambda + psi);

            kernel.at<float>(y + half_height, x + half_width) = gauss * sinusoid;
        }
    }

    // cv::normalize(kernel, kernel, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::filter2D(input_image, filtered_image, -1, kernel);

    return 0;
}

int main(int argc, char *argv[]){

    if (argc<2){
        printf("usage: %s <directory path> <target image path>\n", argv[0]);
    }

    cv::Mat input_image = cv::imread(argv[1]);
    cv::Mat filtered_image;
    int kernelSize = 10;
    double sigma = 8;
    double theta = CV_PI/4;
    double lambda = 5;
    double gamma = 0.8;
    double psi = 0;

    applyGaborFilter(input_image, filtered_image, kernelSize, kernelSize, sigma, theta, lambda, gamma, psi);
    cv::imshow("Original Image", input_image);
    cv::imshow("Filtered Image", filtered_image);
    
    // Wait for key press to exit
    cv::waitKey(0);

    return 0;
}