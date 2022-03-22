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
Mat their_Sobel(Mat source);
Mat our_Sobel(Mat source);
Mat their_Scharr(Mat source);
Mat our_Scharr(Mat source);
Mat their_Canny(Mat source);
Mat our_Canny(Mat source);


int main(int argc, char** argv){
  Mat image, copy, altered_image, opencv_image;
  std::string imageName = "C:\\Users\\pica\\Documents\\GitHub\\super-duper-system\\P2\\pasillo2.pgm";

  image = imread(samples::findFile(imageName), IMREAD_COLOR);
  if(image.empty()){
    printf("Error opening image: %s\n", imageName.c_str());
    return EXIT_FAILURE;
  }

  altered_image = hough(image, 50, 200);

  imshow("their", altered_image);


  //copy = image.clone();
  //altered_image = their_Canny(image);
  //opencv_image = our_Canny(copy);
  
 /* imshow("orig", image);
  imshow("pruebas", altered_image);
  imshow("opencv", opencv_image);*/

  waitKey(0);

  return EXIT_SUCCESS;
}

Mat hough(Mat image, int minThreshold, int maxThreshold){
  Mat blurred_image;
  GaussianBlur(image, blurred_image, Size(5, 5), 0, 0, BORDER_DEFAULT);
  Mat gray_blurred_image;
  cvtColor(blurred_image, gray_blurred_image, COLOR_BGR2GRAY);


  Mat gX = gradX(gray_blurred_image);
  Mat gY = gradY(gray_blurred_image);

  Mat ang = angles(gX, gY);

  Mat canny_result;
  Canny(gray_blurred_image, canny_result, minThreshold, maxThreshold, 3);

  Mat canny_coloured;
  cvtColor(canny_result, canny_coloured, COLOR_GRAY2BGR);

  std::vector<Vec2f> lines;
  HoughLines(canny_result, lines, 1, CV_PI / 180, maxThreshold - minThreshold, 0, 0);
  for (size_t i = 0; i < lines.size(); i++){
    float rho = lines[i][0], theta = lines[i][1];
    Point pt1, pt2;
    double a = cos(theta), b = sin(theta);
    double x0 = a * rho, y0 = b * rho;
    pt1.x = cvRound(x0 + 1000 * (-b));
    pt1.y = cvRound(y0 + 1000 * (a));
    pt2.x = cvRound(x0 - 1000 * (-b));
    pt2.y = cvRound(y0 - 1000 * (a));
    line(canny_coloured, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
  }
  return canny_coloured;
  
}

Mat gradX(Mat gray_blurred_image){
  Mat source = gray_blurred_image.clone();
  Mat cannyX, gradX;
  Mat verticalK = (Mat_<float>(3, 3) <<
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1);

  filter2D(source, cannyX, CV_32F, verticalK);

  convertScaleAbs(cannyX, gradX);

  return gradX;
}

Mat gradY(Mat gray_blurred_image){
  Mat source = gray_blurred_image.clone();
  Mat cannyY, gradY;
  Mat horizontalK = (Mat_<float>(3, 3) <<
    1, 2, 1,
    0, 0, 0,
    -1, -2, -1);

  filter2D(source, cannyY, CV_32F, horizontalK);

  convertScaleAbs(cannyY, gradY);

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
  Mat altered_image, cannyX, angles, cannyY, gradX, gradY, blurred_source, gray_blurred_source;
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
  
  Mat canny_result = Mat(gradX.size(), gradX.type(), 0.0);

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
    }
  }

  imshow("4", angles); // Tendría que imprimir las direcciones (mirar wikipedia de operador canny, sería la imagen 4)

  canny_result.convertTo(canny_result, CV_32F);
  // Non-max supression
  Mat nonMaxSup = Mat(angles.size(), angles.type(), 0.0);
  for (int y = 0; y < angles.rows; y++){
    for (int x = 0; x < angles.cols; x++){
      angles.ptr<float>(y)[x] = angles.ptr<float>(y)[x] * 180 / CV_PI;
      if (angles.ptr<float>(y)[x] < 0){
        angles.ptr<float>(y)[x] = angles.ptr<float>(y)[x] + 180;
      }
    }
  }
  float oneSide, theOtherSide;
  for (int y = 1; y < angles.rows - 1; y++){
    //float* puntY = angles.ptr<float>(y);
    for (int x = 1; x < angles.cols - 1; x++){
      oneSide = 255;
      theOtherSide = 255;
      float angle = angles.ptr<float>(y)[x];
      if (((angle >= 0) && (angle < 22.5)) || ((angle >= 157.5) && (angle <= 180))){
        oneSide = canny_result.ptr<float>(y)[x + 1];
        theOtherSide = canny_result.ptr<float>(y)[x - 1];
      }else if ((angle >= 22.5) && (angle < 67.5)){
        oneSide = canny_result.ptr<float>(y + 1)[x - 1];
        theOtherSide = canny_result.ptr<float>(y - 1)[x + 1];
      }else if ((angle >= 67.5) && (angle < 112.5)){
        oneSide = canny_result.ptr<float>(y + 1)[x];
        theOtherSide = canny_result.ptr<float>(y - 1)[x];
      }else if ((angle >= 112.5) && (angle < 157.5)){
        oneSide = canny_result.ptr<float>(y - 1)[x - 1];
        theOtherSide = canny_result.ptr<float>(y + 1)[x + 1];
      }

      float aux = canny_result.ptr<float>(y)[x];
      if ((aux >= oneSide) && (aux >= theOtherSide)){
        nonMaxSup.ptr<float>(y)[x] = aux;
      }
    }
  }
  //cout << nonMaxSup << endl;

  nonMaxSup.convertTo(nonMaxSup, CV_8U);

  imshow("5", nonMaxSup);

  return altered_image;
}