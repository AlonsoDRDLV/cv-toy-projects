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

Mat hough(Mat image, int minThreshold, int maxThreshold);
Mat gradX(Mat gray_blurred_image);
Mat gradY(Mat gray_blurred_image);
Mat angles(Mat gX, Mat gY);
Mat grad_module(Mat gX, Mat gY);
Mat their_Sobel(Mat source);
Mat our_Sobel(Mat source);
Mat their_Scharr(Mat source);
Mat our_Scharr(Mat source);
Mat their_Canny(Mat source);
Mat our_Canny(Mat source, float minThreshold, float maxThreshold);


int main(int argc, char** argv){
  Mat image, copy, altered_image, opencv_image;
  std::string imageName = "C:\\Users\\pica\\Documents\\GitHub\\super-duper-system\\P2\\poster.pgm";

  image = imread(samples::findFile(imageName), IMREAD_COLOR);
  if(image.empty()){
    printf("Error opening image: %s\n", imageName.c_str());
    return EXIT_FAILURE;
  }

  //altered_image = hough(image, 50, 250);

  //imshow("their", altered_image);


  //copy = image.clone();
  //altered_image = their_Canny(image);
  opencv_image = our_Canny(image, 0.1, 0.3);

  
 /* imshow("orig", image);
  imshow("pruebas", altered_image); */
  //imshow("opencv", opencv_image);

  waitKey(0);

  return EXIT_SUCCESS;
}

std::vector<std::vector<double>> filter_lines(std::vector<Vec4i> lines){
  std::vector<std::vector<double>> filtered_lines;
  int x1, x2, y1, y2;

  Vec4i line;
  for (int i = 0; i < lines.size(); i++){
    line = lines[i];
    x1 = line[0];
    y1 = line[1];
    x2 = line[2];
    y2 = line[3];


  }
  
  return filtered_lines;
}

Mat hough(Mat source, int minThreshold, int maxThreshold){
  Mat canny = our_Canny(source, minThreshold, maxThreshold);
  std::vector<Vec4i> lines;
  HoughLinesP(canny, lines, 1, CV_PI / 180, 50, 15);
  std::vector<std::vector<double>> v = filter_lines(lines);

  return canny;
}

Mat gradX(Mat gray_blurred_image){
  Mat source = gray_blurred_image.clone();
  Mat cannyX, gradX = Mat(source.size(), source.type());
  Mat verticalK = (Mat_<float>(3, 3) <<
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1);

  filter2D(source, gradX, CV_32F, verticalK);

 // convertScaleAbs(cannyX, gradX);

  return gradX;
}

Mat gradY(Mat gray_blurred_image){
  Mat source = gray_blurred_image.clone();
  Mat cannyY, gradY;
  Mat horizontalK = (Mat_<float>(3, 3) <<
    1, 2, 1,
    0, 0, 0,
    -1, -2, -1);

  filter2D(source, gradY, CV_32F, horizontalK);

 // convertScaleAbs(cannyY, gradY);

  return gradY;
}

Mat angles(Mat gX, Mat gY){
  Mat gAuxX = gX.clone();
  Mat gAuxY = gY.clone();

  gAuxX.convertTo(gAuxX, CV_32F);
  gAuxY.convertTo(gAuxY, CV_32F);
  Mat ang = Mat(gAuxX.size(), CV_32F, 0.0);

  for(int y = 0; y < gAuxX.rows; y++){
    for(int x = 0; x < gAuxY.cols; x++){
       ang.at<float>(y, x) = atan2(gAuxX.at<float>(y, x), gAuxY.at<float>(y, x));
    }
  }
  return ang;
}

Mat grad_module(Mat gX, Mat gY){
  Mat gAuxX = gX.clone();
  gAuxX.convertTo(gAuxX, CV_32F);
  pow(gAuxX, 2, gAuxX);
  Mat gAuxY = gY.clone();
  gAuxY.convertTo(gAuxY, CV_32F);
  pow(gAuxY, 2, gAuxY);
  Mat result = Mat(gX.size(), CV_32F);
  result = gAuxY + gAuxX;
  sqrt(result, result);
  float min = result.at<float>(0, 0);
  float max = result.at<float>(0, 0);
  for (int y = 0; y < result.rows; y++){
    for (int x = 0; x < result.cols; x++){
      if(result.at<float>(y, x) < min){
        min = result.at<float>(y, x);
      }
      if(result.at<float>(y, x) > max){
        max = result.at<float>(y, x);
      }
    }
  }
  result = (result - min) / (max - min);
  //cout << result << endl;
  return result;
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

Mat our_Canny(Mat source, float minThreshold, float maxThreshold){
  Mat altered_image, cannyXE, cannyY, gX, gY, blurred_source, gray_blurred_source;
  Mat horizontalK = (Mat_<float>(3, 3) <<
    1, 2, 1,
    0, 0, 0,
    -1, -2, -1);
  Mat verticalK = (Mat_<float>(3, 3) <<
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1);
  imshow("source", source);

  GaussianBlur(source, blurred_source, Size(5, 5), 0, 0, BORDER_DEFAULT);
  cvtColor(blurred_source, gray_blurred_source, COLOR_BGR2GRAY);
  imshow("blurred gray", gray_blurred_source);

  gX = gradX(gray_blurred_source);
  gY = gradY(gray_blurred_source);
  imshow("gradX", gX);
  imshow("gradY", gY);

  Mat canny_module = Mat(gX.size(), gX.type());
  canny_module = grad_module(gX, gY);
  imshow("grad module", canny_module);

  Mat ang = Mat(gX.size(), gX.type());
  ang = angles(gX, gY);

  //cartToPolar(gX, gY, canny_module, ang);
  imshow("angles", ang);

  // Non-max supression
  Mat nonMaxSup = Mat(ang.size(), ang.type(), 0.0);
  for (int y = 0; y < ang.rows; y++){
    for (int x = 0; x < ang.cols; x++){
      ang.at<float>(y, x) = ang.at<float>(y, x) * 180 / CV_PI;
      if (ang.at<float>(y, x) < 0){
        ang.at<float>(y, x) = ang.at<float>(y, x) + 180;
      }
    }
  }
  // cout << ang << endl;

  float oneSide, theOtherSide;
  for (int y = 1; y < ang.rows - 1; y++){
    for (int x = 1; x < ang.cols - 1; x++){
      oneSide = 255;
      theOtherSide = 255;
      float angle = ang.at<float>(y, x);
      if (((angle >= 0) && (angle < 22.5)) || ((angle >= 157.5) && (angle <= 180))){
        oneSide = canny_module.at<float>(y + 1, x);
        theOtherSide = canny_module.at<float>(y - 1, x);
      }else if ((angle >= 22.5) && (angle < 67.5)){
        oneSide = canny_module.at<float>(y + 1, x - 1);
        theOtherSide = canny_module.at<float>(y - 1, x + 1);
      }else if ((angle >= 67.5) && (angle < 112.5)){
        oneSide = canny_module.at<float>(y, x + 1);
        theOtherSide = canny_module.at<float>(y, x - 1);
      }else if ((angle >= 112.5) && (angle < 157.5)){
        oneSide = canny_module.at<float>(y - 1, x - 1);
        theOtherSide = canny_module.at<float>(y + 1, x + 1);
      }

      float aux = canny_module.at<float>(y, x);
      if ((aux >= oneSide) && (aux >= theOtherSide)){
        nonMaxSup.at<float>(y, x) = aux;
      }
    }
  }

  imshow("nonMaxSup", nonMaxSup);

  Mat withThreshold = Mat(nonMaxSup.size(), nonMaxSup.type());
  // Muchos podrían haber caído llegados a este punto; nosotros, no
  for (int y = 0; y < nonMaxSup.rows; y++){
    for (int x = 0; x < nonMaxSup.cols; x++){
      if (nonMaxSup.at<float>(y, x) < minThreshold){
        withThreshold.at<float>(y, x) = 0.0f;
      }else if(nonMaxSup.at<float>(y, x) < maxThreshold){
        withThreshold.at<float>(y, x) = 0.3f;
      }else{
        withThreshold.at<float>(y, x) = 1.0f;
      }
    }
  }

  imshow("with threshold", withThreshold);

  Mat hysteresis = withThreshold.clone();
  for(int y = 1; y < hysteresis.rows - 1; y++){
    for(int x = 1; x < hysteresis.cols - 1; x++){
      if (hysteresis.at<float>(y, x) == 0.3f){
        if ((withThreshold.at<float>(y - 1, x) == 1.0f) ||
            (withThreshold.at<float>(y - 1, x - 1) == 1.0f) || 
            (withThreshold.at<float>(y, x - 1) == 1.0f) || 
            (withThreshold.at<float>(y + 1, x - 1) == 1.0f) || 
            (withThreshold.at<float>(y + 1, x) == 1.0f) || 
            (withThreshold.at<float>(y + 1, x + 1) == 1.0f) || 
            (withThreshold.at<float>(y, x + 1) == 1.0f) || 
            (withThreshold.at<float>(y - 1, x + 1) == 1.0f)){
          hysteresis.at<float>(y, x) = 1.0f;
        }else{
          hysteresis.at<float>(y, x) = 0.0f;
        }
      }
    }
  }

  imshow("hysteresis", hysteresis);
  return hysteresis;
}