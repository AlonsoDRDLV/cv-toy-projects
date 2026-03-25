#include "opencv2/imgcodecs.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <string>
#include <cmath>
#include <vector>

#define CVUI_IMPLEMENTATION
#include "cvui.h"

using std::cin;
using std::cout;
using std::endl;
using namespace cv;


Mat gradX(Mat gray_blurred_image);
Mat gradY(Mat gray_blurred_image);
Mat angles(Mat gX, Mat gY);
Mat grad_module(Mat gX, Mat gY);
Mat their_Sobel(Mat source);
Mat our_Sobel(Mat source, Mat& grad_ang, Mat& grad_mod, int sigma);
Mat their_Scharr(Mat source);
Mat our_Scharr(Mat source, Mat& grad_ang, Mat& grad_mod, int sigma);
Mat their_Canny(Mat source);
Mat our_Canny(Mat source, Mat& grad_ang, Mat& grad_mod, float minThreshold, float maxThreshold, int sigma);
int vanish_point(const Mat& source, Mat& grad_ang, double module_threshold,
  double vertical_threshold, double horizontal_threshold);
Mat drawCross(const Mat& source, int location);

int main(int argc, char** argv) {
  Mat image, copy, altered_image, opencv_image, grad_mod, grad_angle;
  int op = 0, vanishing_point_loc, sigma;
  double module_threshold = 0.8, horizontal_threshold = 0.7, vertical_threshold = 0.3,
    canny_theshold_1 = 0.05, canny_theshold_2 = 0.1;

  std::string imageName = "C:\\Users\\AlonsoDRDLV\\Documents\\GitHub\\super-duper-system\\P2\\pasillo3.pgm";

  image = imread(samples::findFile(imageName), IMREAD_COLOR);
  if (image.empty()) {
    printf("Error opening image: %s\n", imageName.c_str());
    return EXIT_FAILURE;
  }

  cout << "Selecciona el operador:\n 1. Scharr\n 2. Sobel\n 3. Canny" << endl;
  cin >> op;
  cout << endl << "Selecciona el valor de sigma:" << endl;
  cin >> sigma;
  //imshow("their", altered_image);


  copy = image.clone();
  //altered_image = their_Canny(image);
  switch (op) {
  case 1:
    altered_image = our_Scharr(image, grad_angle, grad_mod, sigma);
    break;
  case 2:

    altered_image = our_Sobel(image, grad_angle, grad_mod, sigma);
    break;
  case 3:
    cout << endl << "Selecciona el del threshold 1:" << endl;
    cin >> canny_theshold_1;
    cout << endl << "Selecciona el del threshold 2:" << endl;
    cin >> canny_theshold_2;

    altered_image = our_Canny(image, grad_angle, grad_mod, canny_theshold_1, canny_theshold_2, sigma);
    //altered_image = their_Canny(image);

    cout << endl << "Define el threshold del módulo del gradiente para la traspuesta de Hough" << endl;
    cin >> module_threshold;
    cout << endl << "Define el rango de valores del coseno del angulo del gradiente que se tiene en cuenta: \n limite inferior (filtrado de verticales):" << endl;
    cin >> vertical_threshold;
    cout << endl << "Limite superior (filtrado de horizontales):" << endl;
    cin >> horizontal_threshold;

    vanishing_point_loc = vanish_point(altered_image, grad_angle, module_threshold
      , vertical_threshold, horizontal_threshold);

    drawCross(copy, vanishing_point_loc);

    imshow("pruebas", copy);

    break;
  default:
    cout << "Opcion no definida" << endl;
    return 1;
  }


  waitKey(0);

  return EXIT_SUCCESS;
}

Mat gradX(Mat gray_blurred_image) {
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

Mat gradY(Mat gray_blurred_image) {
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

Mat angles(Mat gX, Mat gY) {
  Mat gAuxX = gX.clone();
  Mat gAuxY = gY.clone();

  gAuxX.convertTo(gAuxX, CV_32F);
  gAuxY.convertTo(gAuxY, CV_32F);
  Mat ang = Mat(gAuxX.size(), CV_32F, 0.0);

  for (int y = 0; y < gAuxX.rows; y++) {
    for (int x = 0; x < gAuxY.cols; x++) {
      ang.at<float>(y, x) = atan2(gAuxX.at<float>(y, x), gAuxY.at<float>(y, x));
    }
  }
  return ang;
}

Mat grad_module(Mat gX, Mat gY) {
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
  for (int y = 0; y < result.rows; y++) {
    for (int x = 0; x < result.cols; x++) {
      if (result.at<float>(y, x) < min) {
        min = result.at<float>(y, x);
      }
      if (result.at<float>(y, x) > max) {
        max = result.at<float>(y, x);
      }
    }
  }
  result = (result - min) / (max - min);
  //cout << result << endl;
  return result;
}

Mat their_Sobel(Mat source) {
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

Mat our_Sobel(Mat source, Mat& grad_ang, Mat& grad_mod, int sigma = 3) {
  Mat altered_image, sobelX, sobelY, gradX, gradY, blurred_source, gray_blurred_source;

  Mat horizontalK = (Mat_<float>(3, 3) <<
    1, 2, 1,
    0, 0, 0,
    -1, -2, -1);
  Mat verticalK = (Mat_<float>(3, 3) <<
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1);

  GaussianBlur(source, blurred_source, Size(5, 5), sigma, sigma, BORDER_DEFAULT);

  cvtColor(blurred_source, gray_blurred_source, COLOR_BGR2GRAY);

  filter2D(gray_blurred_source, sobelX, CV_16S, verticalK);
  filter2D(gray_blurred_source, sobelY, CV_16S, horizontalK);

  convertScaleAbs(sobelX, gradX);
  convertScaleAbs(sobelY, gradY);

  imshow("gradX", gradX);
  imshow("gradY", gradY);

  grad_mod = grad_module(gradX, gradY);

  imshow("grad module", grad_mod);

  grad_ang = angles(gradX, gradY);

  imshow("grad angles", grad_ang);

  addWeighted(gradX, 0.5, gradY, 0.5, 0, altered_image);

  imshow("edges", altered_image);

  return altered_image;
}

Mat their_Scharr(Mat source) {
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

Mat our_Scharr(Mat source, Mat& grad_ang, Mat& grad_mod, int sigma = 3) {
  Mat scharrX, scharrY, gradX, gradY, altered_image, blurred_source, gray_blurred_source;
  Mat horizontalK = (Mat_<float>(3, 3) <<
    3, 10, 3,
    0, 0, 0,
    -3, -10, -3);
  Mat verticalK = (Mat_<float>(3, 3) <<
    -3, 0, 3,
    -10, 0, 10,
    -3, 0, 3);

  GaussianBlur(source, blurred_source, Size(5, 5), sigma, sigma, BORDER_DEFAULT);

  cvtColor(blurred_source, gray_blurred_source, COLOR_BGR2GRAY);

  filter2D(gray_blurred_source, scharrX, CV_16S, verticalK);
  filter2D(gray_blurred_source, scharrY, CV_16S, horizontalK);

  convertScaleAbs(scharrX, gradX);
  convertScaleAbs(scharrY, gradY);

  imshow("gradX", gradX);
  imshow("gradY", gradY);

  grad_mod = grad_module(gradX, gradY);

  imshow("grad module", grad_mod);

  grad_ang = angles(gradX, gradY);

  imshow("grad angles", grad_ang);

  addWeighted(gradX, 0.5, gradY, 0.5, 0, altered_image);

  imshow("edges", altered_image);


  return altered_image;
}

Mat their_Canny(Mat source) {
  Mat altered_image, canny_result, blurred_source, gray_blurred_source;

  GaussianBlur(source, blurred_source, Size(5, 5), 5, 5, BORDER_DEFAULT);

  cvtColor(blurred_source, gray_blurred_source, COLOR_BGR2GRAY);

  Canny(gray_blurred_source, canny_result, 0.1, 0.3, 3);

  altered_image = Scalar::all(0);
  gray_blurred_source.copyTo(altered_image, canny_result);
  imshow("paruebasas", altered_image);
  return altered_image;
}

Mat our_Canny(Mat source, Mat& ang, Mat& canny_module, float minThreshold, float maxThreshold, int sigma = 3) {
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

  GaussianBlur(source, blurred_source, Size(5, 5), sigma, sigma, BORDER_DEFAULT);
  cvtColor(blurred_source, gray_blurred_source, COLOR_BGR2GRAY);
  imshow("blurred gray", gray_blurred_source);

  gX = gradX(gray_blurred_source);
  gY = gradY(gray_blurred_source);
  //imshow("gradX", gX);
  //imshow("gradY", gY);

  //Mat canny_module = Mat(gX.size(), gX.type());
  canny_module = grad_module(gX, gY);
  imshow("grad module", canny_module);

  //Mat ang = Mat(gX.size(), gX.type());
  ang = angles(gX, gY);

  //cartToPolar(gX, gY, canny_module, ang);
  imshow("angles", ang);

  Mat ang_copy = ang.clone();
  // Non-max supression
  Mat nonMaxSup = Mat(ang_copy.size(), ang_copy.type(), 0.0);
  for (int y = 0; y < ang_copy.rows; y++) {
    for (int x = 0; x < ang_copy.cols; x++) {
      ang_copy.at<float>(y, x) = ang_copy.at<float>(y, x) * 180 / CV_PI;
      if (ang_copy.at<float>(y, x) < 0) {
        ang_copy.at<float>(y, x) = ang_copy.at<float>(y, x) + 180;
      }
    }
  }
  // cout << ang << endl;

  float oneSide, theOtherSide;
  for (int y = 1; y < ang_copy.rows - 1; y++) {
    for (int x = 1; x < ang_copy.cols - 1; x++) {
      oneSide = 255;
      theOtherSide = 255;
      float angle = ang_copy.at<float>(y, x);
      if (((angle >= 0) && (angle < 22.5)) || ((angle >= 157.5) && (angle <= 180))) {
        oneSide = canny_module.at<float>(y + 1, x);
        theOtherSide = canny_module.at<float>(y - 1, x);
      }
      else if ((angle >= 22.5) && (angle < 67.5)) {
        oneSide = canny_module.at<float>(y + 1, x - 1);
        theOtherSide = canny_module.at<float>(y - 1, x + 1);
      }
      else if ((angle >= 67.5) && (angle < 112.5)) {
        oneSide = canny_module.at<float>(y, x + 1);
        theOtherSide = canny_module.at<float>(y, x - 1);
      }
      else if ((angle >= 112.5) && (angle < 157.5)) {
        oneSide = canny_module.at<float>(y - 1, x - 1);
        theOtherSide = canny_module.at<float>(y + 1, x + 1);
      }

      float aux = canny_module.at<float>(y, x);
      if ((aux >= oneSide) && (aux >= theOtherSide)) {
        nonMaxSup.at<float>(y, x) = aux;
      }
    }
  }

  imshow("nonMaxSup", nonMaxSup);

  Mat withThreshold = Mat(nonMaxSup.size(), nonMaxSup.type());
  // Muchos podr�an haber ca�do llegados a este punto; nosotros, no
  for (int y = 0; y < nonMaxSup.rows; y++) {
    for (int x = 0; x < nonMaxSup.cols; x++) {
      if (nonMaxSup.at<float>(y, x) < minThreshold) {
        withThreshold.at<float>(y, x) = 0.0f;
      }
      else if (nonMaxSup.at<float>(y, x) < maxThreshold) {
        withThreshold.at<float>(y, x) = 0.3f;
      }
      else {
        withThreshold.at<float>(y, x) = 1.0f;
      }
    }
  }

  imshow("with threshold", withThreshold);

  Mat hysteresis = withThreshold.clone();
  for (int y = 1; y < hysteresis.rows - 1; y++) {
    for (int x = 1; x < hysteresis.cols - 1; x++) {
      if (hysteresis.at<float>(y, x) == 0.3f) {
        if ((withThreshold.at<float>(y - 1, x) == 1.0f) ||
          (withThreshold.at<float>(y - 1, x - 1) == 1.0f) ||
          (withThreshold.at<float>(y, x - 1) == 1.0f) ||
          (withThreshold.at<float>(y + 1, x - 1) == 1.0f) ||
          (withThreshold.at<float>(y + 1, x) == 1.0f) ||
          (withThreshold.at<float>(y + 1, x + 1) == 1.0f) ||
          (withThreshold.at<float>(y, x + 1) == 1.0f) ||
          (withThreshold.at<float>(y - 1, x + 1) == 1.0f)) {
          hysteresis.at<float>(y, x) = 1.0f;
        }
        else {
          hysteresis.at<float>(y, x) = 0.0f;
        }
      }
    }
  }

  imshow("hysteresis", hysteresis);
  return hysteresis;
}

//busca el punto de fuga en la horizontal aplicando la trasformade de Hough
int vanish_point(const Mat& source, Mat& grad_ang, double module_threshold,
  double vertical_threshold, double horizontal_threshold) {

  double rho, theta;
  int x, y, intersection;
  std::vector<int> candidates(source.cols, 0); //pasillo2 tiene 500 algo pixeles de ancho,
                                                //si resulta ser demasiado costoso podríamos clusterizar en
                                                //conjuntos de varios pixeles
  Mat copy = source.clone();
  cvtColor(copy, copy, COLOR_GRAY2BGR);

  for (int i = 0; i < source.cols; i++) {
    for (int j = 0; j < source.rows; j++) {
      if(source.at<float>(j, i) > module_threshold){ //threshold
        //cout << i << " " << j << endl;
        theta = grad_ang.at<float>(j, i); //para mas eficiencia votamos de entre las lineas que pasan por el punto
                                          //las lineas en la dirección del gradiente
        //nos libramos de horizonteles y verticales
        if ((abs(cos(theta)) > vertical_threshold) &&
          (abs(cos(theta)) < horizontal_threshold)) {

          line(copy, Point(i, j), Point(i, j), Scalar(0, 0, 255), 1, LINE_AA);
          x = j - source.rows / 2;
          y = source.cols / 2 - i;
          rho = x * cos(theta) + y * sin(theta);  //ecuación de la recta

          //no nos interesa detectar que lineas son más votadas (cuales son las lineas reales de la imagen)
          //sino con que pixel de la horizontal intersectan más candidatas (se entiende que éste es
          //el punto en el que más lineas de la imagen intersectan y por tanto el punto de fuga)
          intersection = rho / cos(theta); //y = 0 y despejamos x
          intersection = intersection + source.cols / 2; // dejamos el valor en rango 0..cols para indexar el vector
          if (intersection >= 0 && intersection < source.cols) candidates.at(intersection)++;
        }
      }
    }
  }
  int max = 0;
  int i, max_i;
  for (i = 0; i < candidates.size(); i++) {
    if (candidates.at(i) > max) {
      max = candidates.at(i);
      max_i = i;
    }
  }
  imshow("copia", copy);
  //cout << max_i << " " << max << endl;
  //for(int i : candidates) cout << " " << i << " ";
  return max_i;
}

Mat drawCross(const Mat& source, int location) {
  Point p1 = Point(location - 10, source.cols/2);
  Point p2 = Point(location + 10, source.cols / 2);
  line(source, p1, p2, CV_RGB(255, 0, 0), 3);

  p1 = Point(location, source.cols / 2 + 10);
  p2 = Point(location, source.cols / 2 - 10);
  line(source, p1, p2, CV_RGB(255, 0, 0), 3);

  return source;
}
