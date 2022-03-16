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
  Mat image, copy, altered_image, opencv_image;
  std::string imageName = "C:\\Users\\pica\\Documents\\GitHub\\super-duper-system\\P2\\recanny.jpg";

  image = imread(samples::findFile(imageName), IMREAD_COLOR);
  if(image.empty()){
    printf("Error opening image: %s\n", imageName.c_str());
    return EXIT_FAILURE;
  }

  copy = image.clone();

  altered_image = their_Canny(image);
  opencv_image = our_Canny(copy);
  
 /* imshow("orig", image);
  imshow("pruebas", altered_image);
  imshow("opencv", opencv_image);*/

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
  Mat altered_image, scharrX, scharrY, gradX, gradY, blurred_source, gray_blurred_source;
  GaussianBlur(source, blurred_source, Size(3, 3), 0, 0, BORDER_DEFAULT);
  
  cvtColor(blurred_source, gray_blurred_source, COLOR_BGR2GRAY);

  Sobel(gray_blurred_source, scharrX, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
  Sobel(gray_blurred_source, scharrY, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);

  convertScaleAbs(scharrX, gradX);
  convertScaleAbs(scharrY, gradY);
  addWeighted(gradX, 0.5, gradY, 0.5, 0, altered_image);
  return altered_image;
}

Mat our_Sobel(Mat source){
  Mat altered_image, sobelX, sobelY, gradX, gradY, blurred_source, gray_blurred_source;

  Mat horizontalK = (Mat_<float>(3, 3) <<
    1, 2, 1,
    0, 0, 0,
    -1, -2, -1);
  Mat verticalK = (Mat_<float>(3, 3) <<
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1);

  GaussianBlur(source, blurred_source, Size(3, 3), 0, 0, BORDER_DEFAULT);

  cvtColor(blurred_source, gray_blurred_source, COLOR_BGR2GRAY);

  filter2D(gray_blurred_source, sobelX, CV_16S, verticalK);
  filter2D(gray_blurred_source, sobelY, CV_16S, horizontalK);

  convertScaleAbs(sobelX, gradX);
  convertScaleAbs(sobelY, gradY);
  addWeighted(gradX, 0.5, gradY, 0.5, 0, altered_image);
  return altered_image;
}

Mat their_Scharr(Mat source){
  Mat altered_image, scharrX, scharrY, gradX, gradY, blurred_source, gray_blurred_source;

  GaussianBlur(source, blurred_source, Size(3, 3), 0, 0, BORDER_DEFAULT);

  cvtColor(blurred_source, gray_blurred_source, COLOR_BGR2GRAY);

  Scharr(gray_blurred_source, scharrX, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
  Scharr(gray_blurred_source, scharrY, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT);

  convertScaleAbs(scharrX, gradX);
  convertScaleAbs(scharrY, gradY);
  addWeighted(gradX, 0.5, gradY, 0.5, 0, altered_image);
  return altered_image;
}

Mat our_Scharr(Mat source){
  Mat scharrX, scharrY, gradX, gradY, altered_image, blurred_source, gray_blurred_source;
  Mat horizontalK = (Mat_<float>(3, 3) <<
    3, 10, 3,
    0, 0, 0,
    -3, -10, -3);
  Mat verticalK = (Mat_<float>(3, 3) <<
    -3, 0, 3,
    -10, 0, 10,
    -3, 0, 3);

  GaussianBlur(source, blurred_source, Size(3, 3), 0, 0, BORDER_DEFAULT);

  cvtColor(blurred_source, gray_blurred_source, COLOR_BGR2GRAY);

  filter2D(gray_blurred_source, scharrX, CV_16S, verticalK);
  filter2D(gray_blurred_source, scharrY, CV_16S, horizontalK);

  return altered_image;
}

Mat their_Canny(Mat source){
  Mat altered_image, canny_result, blurred_source, gray_blurred_source;

  GaussianBlur(source, blurred_source, Size(5, 5), 0, 0, BORDER_DEFAULT);

  cvtColor(blurred_source, gray_blurred_source, COLOR_BGR2GRAY);

  Canny(gray_blurred_source, canny_result, 0, 0 * 3, 3);

  altered_image = Scalar::all(0);
  gray_blurred_source.copyTo(altered_image, canny_result);
  return altered_image;
}

Mat our_Canny(Mat source){
  Mat altered_image, cannyX, angles, cannyY, gradX, gradY, blurred_source, gray_blurred_source, canny_result;
  Mat horizontalK = (Mat_<float>(3, 3) <<
    1, 2, 1,
    0, 0, 0,
    -1, -2, -1);
  Mat verticalK = (Mat_<float>(3, 3) <<
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1);
    
  imshow("1", source);

  GaussianBlur(source, blurred_source, Size(5, 5), 0, 0, BORDER_DEFAULT);

  cvtColor(blurred_source, gray_blurred_source, COLOR_BGR2GRAY);

  imshow("2", gray_blurred_source);
  
  filter2D(gray_blurred_source, cannyX, CV_32F, verticalK);
  filter2D(gray_blurred_source, cannyY, CV_32F, horizontalK);

  convertScaleAbs(cannyX, gradX);
  convertScaleAbs(cannyY, gradY);
  addWeighted(gradX, 0.5, gradY, 0.5, 0, canny_result);

  imshow("3", canny_result);

  imshow("gradX", gradX);
  imshow("gradY", gradY);

//  phase(gradX, gradY, angles); // En un mundo ideal donde las arcotangentes son fáciles esto funcionaría
  gradX.convertTo(gradX, CV_32F);
  gradY.convertTo(gradY, CV_32F);
  angles = Mat(gradX.size(), gradX.type(), 0.0);
  cout << gradX.rows << " " << gradX.cols << endl;
  for (int y = 0; y < gradX.rows; y++){
    float* punteroX = gradX.ptr<float>(y);
    float* punteroY = gradY.ptr<float>(y);
    float* punteroAngle = angles.ptr<float>(y);
    for (int x = 0; x < gradX.cols; x++){
      float pX = punteroX[x];
      float pY = punteroY[x];
      if (pY == 0){
        punteroAngle[x] = 0;
      }else{
        punteroAngle[x] = atan2(pX, pY); // No sé porqué da ceros, por eso existe el cout de encima y no sirve

      }
      //cout << gradX.at<uchar>(y, x) << endl; //lo imprime fatal, obviamente fatal
      //angles.at<uchar>(y, x) = y % 255; // ESTO FUNCIONA Y ME HA COSTADO HORRORES
    }
  }

  imshow("4", angles); // Tendría que imprimir las direcciones (mirar wikipedia de operador canny, sería la imagen 4)

  altered_image = Scalar::all(0);
  gray_blurred_source.copyTo(altered_image, canny_result);
  return altered_image;
}