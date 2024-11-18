/*
Prem Sukhadwala and Ronak Bhanushali
January 2024
CPP file for the functions being used in the main and histogramMatching files
*/

#include <iostream> //standard input output
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "../include/functions.hpp"
#include "../include/texture.hpp"

using namespace cv;

/*
Given a directory on the command line, scans through the directory for image
files.
Passes the file names to the feature detector function
*/
int writeFeaturesToCSV(char *argv[])
{
  char dirname[256];
  char buffer[256];
  FILE *fp;
  DIR *dirp;
  struct dirent *dp;
  char csv_file_name[256];

  int numFiles = 0;
  char *filePaths[1600];

  // get the directory path
  strcpy(dirname, argv[1]);
  strcpy(csv_file_name, argv[2]);
  printf("Processing directory %s\n", dirname);
  // open the directory
  dirp = opendir(dirname);
  if (dirp == NULL)
  {
    printf("Cannot open directory %s\n", dirname);
    exit(-1);
  }

  // loop over all the files in the image file listing
  while ((dp = readdir(dirp)) != NULL)
  {

    // check if the file is an image
    if (strstr(dp->d_name, ".jpg") ||
        strstr(dp->d_name, ".png") ||
        strstr(dp->d_name, ".ppm") ||
        strstr(dp->d_name, ".tif"))
    {
      printf("processing image file: %s\n", dp->d_name);
      // build the overall filename
      strcpy(buffer, dirname);
      strcat(buffer, "/");
      strcat(buffer, dp->d_name);
      printf("full path name: %s\n", buffer);
      filePaths[numFiles] = strdup(buffer);
      numFiles++;
    }
  }

  /*For each files detected in the directory,
  passes the file to this function for further processing
  */
  for (int i = 0; i < numFiles; i++)
  {
    // printf("My number is: %d\n", i);
    getColorFeatures(csv_file_name, filePaths[i]);
  }

  printf("Terminating first program\n");
  return (0);
}

// detects features and passes the file names and the feature vector to the next function
// will change the function name
int getColorFeatures(char *csv_file_name, char *filePath)
{

  // Read the image
  Mat image = imread(filePath);

  // Check if the image is successfully loaded
  if (image.empty())
  {
    std::cerr << "Error: Couldn't read the image." << std::endl;
    return -1;
  }

  // Get the center of the image
  cv::Point center(image.cols / 2, image.rows / 2);

  // Define the size of the center grid (7x7)
  int grid_size = 7;
  int grid_half_size = grid_size / 2;

  // Define the region of interest (ROI) for the center grid
  cv::Rect roi(center.x - grid_half_size, center.y - grid_half_size, grid_size, grid_size);

  // Extract the center grid
  Mat centerGrid = image(roi).clone(); // Using clone to create a separate copy

  // defining vectors for each color channel
  std::vector<uchar> bVector;
  std::vector<uchar> gVector;
  std::vector<uchar> rVector;

  // makes a vector of size 49 (7x7) for each channel
  for (int i = 0; i < centerGrid.rows; i++)
  {
    for (int j = 0; j < centerGrid.rows; j++)
    {
      uchar B = centerGrid.at<Vec3b>(i, j)[0];
      uchar G = centerGrid.at<Vec3b>(i, j)[1];
      uchar R = centerGrid.at<Vec3b>(i, j)[2];

      bVector.push_back(B);
      gVector.push_back(G);
      rVector.push_back(R);
    }
  }

  // concatenate all the vectors into a single vector
  std::vector<float> Vector(bVector.begin(), bVector.end());
  Vector.insert(Vector.end(), gVector.begin(), gVector.end());
  Vector.insert(Vector.end(), rVector.begin(), rVector.end());

  // passes the vector to the next function for making the csv file
  append_image_data_csv(csv_file_name, filePath, Vector, 0);

  return (0);
}

/*
  Given a filename, and image filename, and the image features, by
  default the function will append a line of data to the CSV format
  file.  If reset_file is true, then it will open the file in 'write'
  mode and clear the existing contents.

  The image filename is written to the first position in the row of
  data. The values in image_data are all written to the file as
  floats.

  The function returns a non-zero value in case of an error.
 */
int append_image_data_csv(char *csv_file_name, char *image_filename, std::vector<float> &image_data, int reset_file)
{
  char buffer[256];
  char mode[8];

  FILE *fp;

  strcpy(mode, "a");

  if (reset_file)
  {
    strcpy(mode, "w");
  }

  fp = fopen(csv_file_name, mode);

  if (!fp)
  {
    // printf("Unable to open output file %s\n", argv[2] );
    std::cout << "Unable to open CSV" << std::endl;
    exit(-1);
  }

  // write the filename and the feature vector to the CSV file
  // std::cout << image_filename << std::endl;
  strcpy(buffer, image_filename);
  std::fwrite(buffer, sizeof(char), strlen(buffer), fp);
  for (int i = 0; i < image_data.size(); i++)
  {
    char tmp[256];
    sprintf(tmp, ",%.4f", image_data[i]);
    std::fwrite(tmp, sizeof(char), strlen(tmp), fp);
  }

  std::fwrite("\n", sizeof(char), 1, fp); // EOL

  fclose(fp);
  return (0);
}

/*
  reads a string from a CSV file. the 0-terminated string is returned in the char array os.

  The function returns false if it is successfully read. It returns true if it reaches the end of the line or the file.
 */

int getstring(FILE *fp, char os[])
{
  int p = 0;
  int eol = 0;

  for (;;)
  {
    char ch = fgetc(fp);
    if (ch == ',')
    {
      break;
    }
    else if (ch == '\n' || ch == EOF)
    {
      eol = 1;
      break;
    }
    // printf("%c", ch ); // uncomment for debugging
    os[p] = ch;
    p++;
  }
  // printf("\n"); // uncomment for debugging
  os[p] = '\0';

  return (eol); // return true if eol
}

int getint(FILE *fp, int *v)
{
  char s[256];
  int p = 0;
  int eol = 0;

  for (;;)
  {
    char ch = fgetc(fp);
    if (ch == ',')
    {
      break;
    }
    else if (ch == '\n' || ch == EOF)
    {
      eol = 1;
      break;
    }

    s[p] = ch;
    p++;
  }
  s[p] = '\0'; // terminator
  *v = atoi(s);

  return (eol); // return true if eol
}

/*
  Utility function for reading one float value from a CSV file

  The value is stored in the v parameter

  The function returns true if it reaches the end of a line or the file
 */
int getfloat(FILE *fp, float *v)
{
  char s[256];
  int p = 0;
  int eol = 0;

  for (;;)
  {
    char ch = fgetc(fp);
    if (ch == ',')
    {
      break;
    }
    else if (ch == '\n' || ch == EOF)
    {
      eol = 1;
      break;
    }

    s[p] = ch;
    p++;
  }
  s[p] = '\0'; // terminator
  *v = atof(s);

  return (eol); // return true if eol
}

// Function to calculate target image feature vector
std::vector<float> targetColorFeatures(char *filePath)
{

  Mat image = imread(filePath);

  // Check if the image is successfully loaded
  if (image.empty())
  {
    std::cerr << "Error: Couldn't read the image." << std::endl;
  }

  // Get the center of the image
  cv::Point center(image.cols / 2, image.rows / 2);

  // Define the size of the center grid (7x7)
  int grid_size = 7;
  int grid_half_size = grid_size / 2;

  // Define the region of interest (ROI) for the center grid
  cv::Rect roi(center.x - grid_half_size, center.y - grid_half_size, grid_size, grid_size);

  // Extract the center grid
  Mat centerGrid = image(roi).clone(); // Using clone to create a separate copy

  // defining vectors for each color channel
  std::vector<uchar> bVector;
  std::vector<uchar> gVector;
  std::vector<uchar> rVector;

  // makes a vector of size 49 (7x7) for each channel
  for (int i = 0; i < centerGrid.rows; i++)
  {
    for (int j = 0; j < centerGrid.rows; j++)
    {
      uchar B = centerGrid.at<Vec3b>(i, j)[0];
      uchar G = centerGrid.at<Vec3b>(i, j)[1];
      uchar R = centerGrid.at<Vec3b>(i, j)[2];

      bVector.push_back(B);
      gVector.push_back(G);
      rVector.push_back(R);
    }
  }

  // concatenate all the vectors into a single vector
  std::vector<float> targetVector(bVector.begin(), bVector.end());
  targetVector.insert(targetVector.end(), gVector.begin(), gVector.end());
  targetVector.insert(targetVector.end(), rVector.begin(), rVector.end());

  printf("Target feature vector created\n");

  return targetVector;
}

// Function to compare the feature vector of the target image with the feature vectors in the CSV file
int compareFeatures(char *argv[])
{
  char imagePath[256];
  char csvFileName[256];
  strcpy(imagePath, argv[3]);
  strcpy(csvFileName, argv[2]);
  std::vector<char *> filenames;
  std::vector<std::vector<float>> data;
  int echo = 0;
  std::vector<float> targetVector = targetColorFeatures(imagePath);
  std::vector<std::pair<float, std::string>> image_ranks; // defining a vector pair (float,string)

  // read the csv file
  int result = read_image_data_csv(csvFileName, filenames, data, echo);

  if (result == 0)
  {
    // retreiving filenames and data from the csv file
    for (int i = 0; i < data.size(); i++)
    {
      float temp = 0;
      for (int j = 0; j < data[i].size(); j++)
      {
        // calculating SSD
        float diff = data[i][j] - targetVector[j];
        temp += diff * diff;
      }
      // taking square root
      float dist = std::sqrt(temp);

      std::string current_file_name = filenames[i];
      image_ranks.push_back(std::make_pair(dist, current_file_name));
    }

    // sorting the vector pair in ascending order of the float values
    sort(image_ranks.begin(), image_ranks.end());

    // display images
    cv::Mat target_image = cv::imread(imagePath);
    cv::Mat disp_image_1 = cv::imread(image_ranks[1].second);
    cv::Mat disp_image_2 = cv::imread(image_ranks[2].second);
    cv::Mat disp_image_3 = cv::imread(image_ranks[3].second);
    std::cout<<image_ranks[1].second<<std::endl;
    std::cout<<image_ranks[2].second<<std::endl;
    std::cout<<image_ranks[3].second<<std::endl;

    cv::imshow("Best Match 1",disp_image_1);
    cv::imshow("Best Match 2",disp_image_2);
    cv::imshow("Best Match 3",disp_image_3);
    cv::imshow("Target Image",target_image);
    cv::waitKey(0);
    // Free allocated memory in the filenames vector
    for (char *fname : filenames)
    {
      delete[] fname;
    }
  }

  else
  {
    std::cerr << "Error reading CSV file.\n";
  }
  printf("Terminating second program\n");
  return (0);
}

// Function to compare the feature vector of the target image with the ResNet18 512 embedded vector
int compareEmbeddedVectors(char *argv[])
{
  char imageName[256];
  char csvFileName[256];
  strcpy(imageName, argv[3]);
  strcpy(csvFileName, argv[2]);
  std::vector<char *> filenames;
  std::vector<std::vector<float>> data;
  int echo = 0;
  std::vector<float> target_vector;
  std::vector<std::pair<float, std::string>> image_ranks; // defining a vector pair (float,string)

  // read the csv file
  int result = read_image_data_csv(csvFileName, filenames, data, echo);

  if (result == 0)
  {
    // retreiving filenames and data from the csv file
    printf("Starting task 5\n");

    for (int k = 0; k < data.size(); k++)
    {
      std::string currentFile = filenames[k];
      if (currentFile == imageName)
      {
        target_vector = data[k];
      }
    }
    // printf("test\n");
    for (int i = 0; i < data.size(); i++)
    {
      float temp = 0;
      for (int j = 0; j < data[i].size(); j++)
      {
        // calculating SSD
        float diff = data[i][j] - target_vector[j];
        temp += diff * diff;
      }
      // taking square root
      float dist = std::sqrt(temp);
      // printf("Processing...\n");
      std::string current_file_name = filenames[i];

      image_ranks.push_back(std::make_pair(dist, current_file_name));
    }

    // sorting the vector pair in ascending order of the float values
    sort(image_ranks.begin(), image_ranks.end());

    // for (int i = 0; i < image_ranks.size(); i++)
    // {
    //   std::cout << image_ranks[i].first << std::endl;
    //   std::cout << image_ranks[i].second << std::endl;
    // }
    printf("Best match 1:");
    std::cout << " " << image_ranks[1].second << std::endl;

    printf("Best match 2:");
    std::cout << " " << image_ranks[2].second << std::endl;

    printf("Best match 3:");
    std::cout << " " << image_ranks[3].second << std::endl;

    // Free allocated memory in the filenames vector
    for (char *fname : filenames)
    {
      delete[] fname;
    }
    printf("test3\n");
  }

  else
  {
    std::cerr << "Error reading CSV file.\n";
  }
  printf("Terminating second program\n");
  return (0);
}

/*
  Given a file with the format of a string as the first column and
  floating point numbers as the remaining columns, this function
  returns the filenames as a std::vector of character arrays, and the
  remaining data as a 2D std::vector<float>.

  filenames will contain all of the image file names.
  data will contain the features calculated from each image.

  If echo_file is true, it prints out the contents of the file as read
  into memory.

  The function returns a non-zero value if something goes wrong.
 */
int read_image_data_csv(char *filename, std::vector<char *> &filenames, std::vector<std::vector<float>> &data, int echo_file)
{
  FILE *fp;
  float fval;
  char img_file[256];

  fp = fopen(filename, "r");
  if (!fp)
  {
    printf("Unable to open feature file\n");
    return (-1);
  }

  printf("Reading %s\n", filename);
  for (;;)
  {
    std::vector<float> dvec;

    // read the filename
    if (getstring(fp, img_file))
    {
      break;
    }

    // read the whole feature file into memory
    for (;;)
    {
      // get next feature
      float eol = getfloat(fp, &fval);
      dvec.push_back(fval);
      if (eol)
        break;
    }

    data.push_back(dvec);

    char *fname = new char[strlen(img_file) + 1];
    strcpy(fname, img_file);
    filenames.push_back(fname);
  }
  fclose(fp);
  printf("Finished reading CSV file\n");

  if (echo_file)
  {
    for (int i = 0; i < data.size(); i++)
    {
      for (int j = 0; j < data[i].size(); j++)
      {
        printf("%.4f  ", data[i][j]);
      }
      printf("\n");
    }
    printf("\n");
  }

  return (0);
}

// Generates RG chromaticity histogram of a given image
int generateHistogramRG(cv::Mat &img, cv::Mat &hist)
{

  // Define number of bins
  int num_bins = 16;

  //Empty histogram images full of zeros
  hist = cv::Mat::zeros(cv::Size(num_bins, num_bins), CV_32FC1);

  for (int i = 0; i < img.rows; i++)
  {
    for (int j = 0; j < img.cols; j++)
    {

      //Get BGR values
      float B = img.at<cv::Vec3b>(i, j)[0];
      float G = img.at<cv::Vec3b>(i, j)[1];
      float R = img.at<cv::Vec3b>(i, j)[2];

      //Denominator taken as sum of three to find chromaticity
      float divisor = R + G + B;

      //Avoiding case where denominator is zero
      divisor = divisor > 0.0 ? divisor : 1.0;

      //Get RG values
      float r = R / divisor;
      float g = G / divisor;

      //Find the right bin to put the value
      int rindex = (int)(r * (num_bins - 1) + 0.5);
      int gindex = (int)(g * (num_bins - 1) + 0.5);

      //Add values to corresponding bins
      hist.at<float>(rindex, gindex)++;
    }
  }

  //Number of pixels
  int num_pixels = img.rows * img.cols;

  //Divide by number of pixels so that total area is 1
  hist /= num_pixels;
  return 0;
}

//Generates HS (HSV space) histogram of the given image 
int generateHistogramHS(cv::Mat &img, cv::Mat &hist)
{
  // Define number of bins
  int num_bins = 16;
  
  //Empty histogram images full of zeros
  hist = cv::Mat::zeros(cv::Size(num_bins, num_bins), CV_32FC1);

  //Convert to HSV
  cv::Mat hsv_image;
  cv::cvtColor(img, hsv_image, cv::COLOR_BGR2HSV);

  for (int i = 0; i < hsv_image.rows; i++)
  {
    for (int j = 0; j < hsv_image.cols; j++)
    {

      //Divide h value by 180 to get 0-1 value since opencv maps h values to 0-180 to fit in uchar
      float h = hsv_image.at<cv::Vec3b>(i, j)[0] / 180.0;

      //Divide s value with 255 to get 0-1 value
      float s = hsv_image.at<cv::Vec3b>(i, j)[1] / 255.0;

      //Find the right bin
      int h_index = static_cast<int>(h * (num_bins - 1));
      int s_index = static_cast<int>(s * (num_bins - 1));

      //Add to the right bin
      hist.at<float>(h_index, s_index)++;
    }
  }

  //Normalize area
  int num_pixels = img.rows * img.cols;

  hist /= num_pixels;

  return 0;
}

float histIntersectionArea(cv::Mat &hist1, cv::Mat &hist2)
{
  // Get number of rows and columns
  int num_rows = hist1.rows;
  int num_cols = hist2.cols;
  // Initialize intersection area
  float total_area = 0;

  // Iterate through the histograms
  for (int i = 0; i < num_rows; i++)
  {
    for (int j = 0; j < num_cols; j++)
    {
      float area = min(hist1.at<float>(i, j), hist2.at<float>(i, j)); // Intersection is essentially the lower value
      // Add to total intersection area
      total_area += area;
    }
  }
  return total_area;
}

int histogramMatching(char *argv[])
{
  /** Get files locations of images as done in task 1 **/

  char dirname[256];
  char buffer[256];
  FILE *fp;
  DIR *dirp;
  struct dirent *dp;

  int numFiles = 0;
  char *filePaths[1600];

  // get the directory path
  strcpy(dirname, argv[1]);
  printf("Processing directory %s\n", dirname);
  // open the directory
  dirp = opendir(dirname);
  if (dirp == NULL)
  {
    printf("Cannot open directory %s\n", dirname);
    exit(-1);
  }

  // loop over all the files in the image file listing
  while ((dp = readdir(dirp)) != NULL)
  {

    // check if the file is an image
    if (strstr(dp->d_name, ".jpg") ||
        strstr(dp->d_name, ".png") ||
        strstr(dp->d_name, ".ppm") ||
        strstr(dp->d_name, ".tif"))
    {
      printf("processing image file: %s\n", dp->d_name);
      // build the overall filename
      strcpy(buffer, dirname);
      strcat(buffer, "/");
      strcat(buffer, dp->d_name);
      printf("full path name: %s\n", buffer);
      filePaths[numFiles] = strdup(buffer);
      numFiles++;
    }
  }

  //get target image from args
  char target_image_path[256];
  strcpy(target_image_path, argv[2]);
  
  //Initialize vector containing pairs to hold intersection area and image name. Check both histograms to see if they give the same images or not
  std::vector<std::pair<float, std::string>> image_ranks_hs;
  std::vector<std::pair<float, std::string>> image_ranks_rg;

  //Read target image and get histograms
  cv::Mat target_image = cv::imread(target_image_path);
  cv::Mat target_image_histogram_hs;
  cv::Mat target_image_histogram_rg;
  generateHistogramHS(target_image, target_image_histogram_hs);
  generateHistogramRG(target_image, target_image_histogram_rg);

  //Go through all images
  for (int i = 0; i < numFiles; i++)
  {
    //Read image to check and generate histograms
    cv::Mat query_image = cv::imread(filePaths[i]);
    cv::Mat query_histogram_hs;
    cv::Mat query_histogram_rg;
    generateHistogramHS(query_image, query_histogram_hs);
    generateHistogramRG(query_image, query_histogram_rg);

    //Get overlapping areas
    float overlapping_area_hs = histIntersectionArea(target_image_histogram_hs, query_histogram_hs);
    float overlapping_area_rg = histIntersectionArea(target_image_histogram_rg, query_histogram_rg);

    //Save to vector with corresponding file names
    image_ranks_hs.push_back(std::make_pair(overlapping_area_hs, filePaths[i]));
    image_ranks_rg.push_back(std::make_pair(overlapping_area_rg, filePaths[i]));
  }
  // Sorting in descending order
  sort(image_ranks_hs.rbegin(), image_ranks_hs.rend());
  sort(image_ranks_rg.rbegin(), image_ranks_rg.rend());

  //Display all images
  cv::Mat disp_image_1 = cv::imread(image_ranks_hs[0].second);
  cv::Mat disp_image_2 = cv::imread(image_ranks_hs[1].second);
  cv::Mat disp_image_3 = cv::imread(image_ranks_hs[2].second);
  cv::Mat disp_image_4 = cv::imread(image_ranks_rg[0].second);
  cv::Mat disp_image_5 = cv::imread(image_ranks_rg[1].second);
  cv::Mat disp_image_6 = cv::imread(image_ranks_rg[2].second);
  cv::imshow("HS Best Match 1", disp_image_1);
  cv::imshow("HS Best Match 2", disp_image_2);
  cv::imshow("HS Best Match 3", disp_image_3);
  cv::imshow("Target Image", target_image);
  cv::imshow("RG Best Match 1", disp_image_4);
  cv::imshow("RG Best Match 2", disp_image_5);
  cv::imshow("RG Best Match 3", disp_image_6);
  cv::waitKey(0);

  return 0;
}

int multiHistogramMatching(char *argv[])
{
  /** Get files locations of images as done in task 1 **/
  char dirname[256];
  char buffer[256];
  FILE *fp;
  DIR *dirp;
  struct dirent *dp;

  int numFiles = 0;
  char *filePaths[1600];

  // get the directory path
  strcpy(dirname, argv[1]);
  printf("Processing directory %s\n", dirname);
  // open the directory
  dirp = opendir(dirname);
  if (dirp == NULL)
  {
    printf("Cannot open directory %s\n", dirname);
    exit(-1);
  }

  // loop over all the files in the image file listing
  while ((dp = readdir(dirp)) != NULL)
  {

    // check if the file is an image
    if (strstr(dp->d_name, ".jpg") ||
        strstr(dp->d_name, ".png") ||
        strstr(dp->d_name, ".ppm") ||
        strstr(dp->d_name, ".tif"))
    {
      printf("processing image file: %s\n", dp->d_name);
      // build the overall filename
      strcpy(buffer, dirname);
      strcat(buffer, "/");
      strcat(buffer, dp->d_name);
      printf("full path name: %s\n", buffer);
      filePaths[numFiles] = strdup(buffer);
      numFiles++;
    }
  }

  //get target image name from args
  char target_image_path[256];
  strcpy(target_image_path, argv[2]);

  //Initialize pairs of vectors
  std::vector<std::pair<float, std::string>> image_ranks;

  //Read target image
  cv::Mat target_image = cv::imread(target_image_path);

  //get center of image to get center roi
  int centerX = target_image.cols / 2;
  int centerY = target_image.rows / 2;

  //Get a 150 by 150 roi from center to match
  int roi_width = 150;
  int roi_height = 150;

  //get roi center
  int roiX = centerX - roi_width / 2;
  int roiY = centerY - roi_height / 2;

  //Extract target roi
  cv::Rect roiRect(roiX, roiY, roi_width, roi_height);
  cv::Mat roi = target_image(roiRect).clone();

  //get hostograms for image and target
  cv::Mat target_center_histogram;
  cv::Mat target_image_histogram;
  generateHistogramHS(target_image, target_image_histogram);
  generateHistogramHS(roi, target_center_histogram);

  //go through all images
  for (int i = 0; i < numFiles; i++)
  {
    //Get ROI and histograms for image in question
    cv::Mat query_image = cv::imread(filePaths[i]);
    cv::Mat query_roi = query_image(roiRect).clone();
    cv::Mat query_histogram;
    cv::Mat query_center_histogram;
    generateHistogramHS(query_image, query_histogram);
    generateHistogramHS(query_roi,query_center_histogram);

    //Get area overlap
    float overlapping_area_image = histIntersectionArea(target_image_histogram, query_histogram);
    float overlapping_area_center = histIntersectionArea(target_center_histogram,query_center_histogram);

    //weighted sum
    float weighted_area_sum = 0.7 * overlapping_area_center + 0.3 * overlapping_area_image;

    //Add to vector with file name
    image_ranks.push_back(std::make_pair(weighted_area_sum,filePaths[i]));

    //Print files remaining to not keep user in the dark about how long the program is going to take to execute
    int files_remaining = numFiles - i;
    std::cout << files_remaining << " Files remaining";
  }

  //Sort to get best results
  sort(image_ranks.rbegin(), image_ranks.rend());

  cv::Mat disp_image_1 = cv::imread(image_ranks[0].second);
  cv::Mat disp_image_2 = cv::imread(image_ranks[1].second);
  cv::Mat disp_image_3 = cv::imread(image_ranks[2].second);
  cv::imshow("Best Match 1", disp_image_1);
  cv::imshow("Best Match 2", disp_image_2);
  cv::imshow("Best Match 3", disp_image_3);
  cv::imshow("Target Image", target_image);
  cv::waitKey(0);
  return 0;
}

int customHistMatching(char* argv[]) {
  char dirname[256];
  char buffer[256];
  FILE *fp;
  DIR *dirp;
  struct dirent *dp;

  int numFiles = 0;
  char *filePaths[1600];

  // get the directory path
  strcpy(dirname, argv[1]);
  printf("Processing directory %s\n", dirname);
  // open the directory
  dirp = opendir(dirname);
  if (dirp == NULL)
  {
    printf("Cannot open directory %s\n", dirname);
    exit(-1);
  }

  // loop over all the files in the image file listing
  while ((dp = readdir(dirp)) != NULL)
  {

    // check if the file is an image
    if (strstr(dp->d_name, ".jpg") ||
        strstr(dp->d_name, ".png") ||
        strstr(dp->d_name, ".ppm") ||
        strstr(dp->d_name, ".tif"))
    {
      printf("processing image file: %s\n", dp->d_name);
      // build the overall filename
      strcpy(buffer, dirname);
      strcat(buffer, "/");
      strcat(buffer, dp->d_name);
      printf("full path name: %s\n", buffer);
      filePaths[numFiles] = strdup(buffer);
      numFiles++;
    }
  }
  char target_image_path[256];
  strcpy(target_image_path, argv[2]);

  //defining paired vector for sorting
  std::vector<std::pair<float, int>> hist_ranks;
  std::vector<std::pair<float, std::string>> image_ranks;


  cv::Mat target_image = cv::imread(target_image_path);

  //calculating the centre of the image
  int centerX = target_image.cols / 2;
  int centerY = target_image.rows / 2;

  //defining the ROI dimensions
  int roi_width = target_image.cols / 2;
  int roi_height = target_image.rows / 2;

  //centre
  int roiX = centerX - roi_width / 2;
  int roiY = centerY - roi_height / 2;

  //defining 5 areas of interest
  cv::Rect centreRect(roiX, roiY, roi_width, roi_height);
  cv::Rect topLeftRect(0, 0, roi_width, roi_height);
  cv::Rect topRightRect(centerX, 0, roi_width, roi_height);
  cv::Rect bottomLeftRect(0, centerY, roi_width, roi_height);
  cv::Rect bottomRightRect(centerX, centerY, roi_width, roi_height);

  std::vector<cv::Rect> roi_vector = {centreRect, topLeftRect, topRightRect, bottomLeftRect, bottomRightRect};

  cv::Mat sx;
  cv::Mat sy;
  cv::Mat target_mag = target_image(centreRect).clone();
  cv::Mat centreRoi = target_image(centreRect).clone();
  cv::Mat target_center_histogram;
  cv::Mat target_mag_histogram;
  
  //applying magnitude filter on target image
  sobelX3x3(target_mag,sx);
  sobelY3x3(target_mag,sy);
  magnitude(sx, sy ,target_mag);

  //generate target magnitude and color histograms
  generateHistogramRG(target_mag, target_mag_histogram);
  generateHistogramRG(centreRoi, target_center_histogram);

  for (int i = 0; i < numFiles; i++){
    cv::Mat query_image = cv::imread(filePaths[i]);

    //finding the ROI with the highest intersection with target image (centre ROI)
    for(int j = 0; j<roi_vector.size();j++){
      cv::Mat query_roi = query_image(roi_vector[j]).clone();
      cv::Mat query_histogram;
      generateHistogramRG(query_roi, query_histogram);
      float overlap_area = histIntersectionArea(target_center_histogram, query_histogram);
      hist_ranks.push_back(std::make_pair(overlap_area, j));
    }
    //sorting in descending order
    sort(hist_ranks.rbegin(), hist_ranks.rend());

    cv::Mat query_sx;
    cv::Mat query_sy;
    cv::Mat query_mag = query_image(roi_vector[hist_ranks[0].second]).clone();

    //applying magnitude filter on query image
    sobelX3x3(query_mag, query_sx);
    sobelY3x3(query_mag, query_sy);
    magnitude(query_sx, query_sy, query_mag);

    cv::Mat query_mag_histogram;

    //generate magnitude histogram
    generateHistogramRG(query_mag, query_mag_histogram);

    //calculate intersection area
    float overlapping_area_sobel = histIntersectionArea(target_mag_histogram, query_mag_histogram);
    float weighted_sum = 0.3*overlapping_area_sobel + 0.7*hist_ranks[0].first;    //weighted sum

    image_ranks.push_back(std::make_pair(weighted_sum,filePaths[i]));
    
  }
  //sorting the paired vector in descending order
  sort(image_ranks.rbegin(), image_ranks.rend());

  //display results
  cv::Mat disp_image_1 = cv::imread(image_ranks[1].second);
  cv::Mat disp_image_2 = cv::imread(image_ranks[2].second);
  cv::Mat disp_image_3 = cv::imread(image_ranks[3].second);
  cv::Mat disp_image_4 = cv::imread(image_ranks[4].second);
  cv::Mat disp_image_5 = cv::imread(image_ranks[5].second);

  std::cout<<image_ranks[1].second<<std::endl;
  std::cout<<image_ranks[2].second<<std::endl;
  std::cout<<image_ranks[3].second<<std::endl;
  std::cout<<image_ranks[4].second<<std::endl;
  std::cout<<image_ranks[5].second<<std::endl;
  
  cv::imshow("Best Match 1",disp_image_1);
  cv::imshow("Best Match 2",disp_image_2);
  cv::imshow("Best Match 3",disp_image_3);
  cv::imshow("Best Match 4",disp_image_4);
  cv::imshow("Best Match 5",disp_image_5);
  cv::imshow("Target Image",target_image);
  
  cv::waitKey(0);
  return 0;
}

