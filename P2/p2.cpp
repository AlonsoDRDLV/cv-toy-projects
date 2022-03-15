#include "opencv2/imgcodecs.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <string>
#include <cmath>

#define CVUI_IMPLEMENTATION
#include "cvui.h"

using std::cin;
using std::cout;
using std::endl;
using namespace cv;

int isWindowOpen(const cv::String& name);
void openWindow(const cv::String & name);
void closeWindow(const cv::String& name);
Mat their_Sobel(Mat source);
Mat our_Sobel(Mat source);
Mat their_Scharr(Mat source);
Mat our_Scharr(Mat source);
Mat their_Canny(Mat source);
Mat our_Canny(Mat source);


int main(int argc, char** argv){

  cout << "\nPress 'ESC' to exit program.\nPress 'R' to reset values ( ksize will be -1 equal to Scharr function )";
  // First we declare the variables we are going to use
  Mat image, src, src_gray;
  Mat grad;
  int ksize = 3;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;
  std::string imageName = "C:\\Users\\pica\\Desktop\\CUARTO\\CUATRI2\\VC\\PRACTICAS\\P2\\Contornos\\poster.pgm";
  // As usual we load our source image (src)
  image = imread(samples::findFile(imageName), IMREAD_COLOR); // Load an image
  // Check if image is loaded fine
  if(image.empty()){
    printf("Error opening image: %s\n", imageName.c_str());
    return EXIT_FAILURE;
  }

  // Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
  GaussianBlur(image, src, Size(3, 3), 0, 0, BORDER_DEFAULT);
  // Convert the image to grayscale
  cvtColor(src, src_gray, COLOR_BGR2GRAY);

  Mat altered_image, opencv_image;

  altered_image = their_Canny(src_gray);

  opencv_image = their_Canny(src_gray);
  
  imshow("orig", image);
  imshow("pruebas", altered_image);
  imshow("opencv", opencv_image);

  waitKey(0);

  return EXIT_SUCCESS;
}

int isWindowOpen(const cv::String& name){
  return cv::getWindowProperty(name, cv::WND_PROP_AUTOSIZE) != -1;
}

void openWindow(const cv::String& name){
  cv::namedWindow(name);
  cvui::watch(name);
}

void closeWindow(const cv::String& name){
  cv::destroyWindow(name);

  cv::waitKey(1);
}

Mat their_Sobel(Mat source){
  Mat altered_image, scharrX, scharrY, gradX, gradY;
  Sobel(source, scharrX, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
  Sobel(source, scharrY, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);

  convertScaleAbs(scharrX, gradX);
  convertScaleAbs(scharrY, gradY);
  addWeighted(gradX, 0.5, gradY, 0.5, 0, altered_image);
  return altered_image;
}

Mat our_Sobel(Mat source){
  Mat sobelX, sobelY, gradX, gradY, altered_image;
  Mat horizontalK = (Mat_<float>(3, 3) <<
    1, 2, 1,
    0, 0, 0,
    -1, -2, -1);
  Mat verticalK = (Mat_<float>(3, 3) <<
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1);

  filter2D(source, sobelX, CV_64F, verticalK);
  filter2D(source, sobelY, CV_64F, horizontalK);

  convertScaleAbs(sobelX, gradX);
  convertScaleAbs(sobelY, gradY);
  addWeighted(gradX, 0.5, gradY, 0.5, 0, altered_image);
  return altered_image;
}

Mat their_Scharr(Mat source){
  Mat altered_image, scharrX, scharrY, gradX, gradY;
  Scharr(source, scharrX, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
  Scharr(source, scharrY, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT);

  convertScaleAbs(scharrX, gradX);
  convertScaleAbs(scharrY, gradY);
  addWeighted(gradX, 0.5, gradY, 0.5, 0, altered_image);
  return altered_image;
}

Mat our_Scharr(Mat source){
  Mat scharrX, scharrY, gradX, gradY, altered_image;
  Mat horizontalK = (Mat_<float>(3, 3) <<
    3, 10, 3,
    0, 0, 0,
    -3, -10, -3);
  Mat verticalK = (Mat_<float>(3, 3) <<
    -3, 0, 3,
    -10, 0, 10,
    -3, 0, 3);

  filter2D(source, scharrX, CV_64F, verticalK);
  filter2D(source, scharrY, CV_64F, horizontalK);

  return altered_image;
}

Mat their_Canny(Mat source){
  Mat altered_image, detected_edges;
  blur(source, detected_edges, Size(3, 3));
  Canny(detected_edges, detected_edges, 0, 0 * 3, 3);

  altered_image = Scalar::all(0);
  source.copyTo(altered_image, detected_edges);
  return altered_image;
}

Mat our_Canny(Mat source){
  Mat altered_image, cannyX, cannyY, gradX, gradY, detected_edges;
  Mat horizontalK = (Mat_<float>(3, 3) <<
    1, 2, 1,
    0, 0, 0,
    -1, -2, -1);
  Mat verticalK = (Mat_<float>(3, 3) <<
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1);

  filter2D(source, cannyX, CV_64F, verticalK);
  filter2D(source, cannyY, CV_64F, horizontalK);

  convertScaleAbs(cannyX, gradX);
  convertScaleAbs(cannyY, gradY);
  addWeighted(gradX, 0.5, gradY, 0.5, 0, altered_image);

  altered_image = Scalar::all(0);
  source.copyTo(altered_image, detected_edges);
  return altered_image;
}