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

Mat ourSobel(Mat source){
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
  return sobelX;
}

int main(int argc, char** argv){

  cout << "\nPress 'ESC' to exit program.\nPress 'R' to reset values ( ksize will be -1 equal to Scharr function )";
  // First we declare the variables we are going to use
  Mat image, src, src_gray;
  Mat grad;
  const String window_name = "Pruebas";
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
  Mat grad_x, grad_y, altered_image;
  Mat abs_grad_x, abs_grad_y;
  // derivatives towards x coordinates XOrder = 1, YOrder = 0
  altered_image = ourSobel(src_gray);
  // derivatives towards y coordinates XOrder = 0, YOrder = 1
  //Sobel(src_gray, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
  // converting back to CV_8U
  //convertScaleAbs(grad_x, altered_image);
  //convertScaleAbs(grad_y, abs_grad_y);
  //addWeighted(altered_image, 0.5, abs_grad_y, 0.5, 0, grad);
  
  imshow("orig", image);
  imshow(window_name, altered_image);

  waitKey(0);

  return EXIT_SUCCESS;
}

//int main(int argc, char** argv) {
//  VideoCapture capture;
//  capture.open(0);
//  CascadeClassifier face_cascade;
//  Mat image;
//  double alpha = 1; /*< Simple contrast control */
//  int option = -1, div, n_colors = 3, blur_factor = 5, depth;
//  double blue = 0.0, green = 0.0, red = 0.0;
//  float factorX, factorY;
//  int aux1, aux2, aux3;
//  aux1 = aux2 = aux3 = 1;
//  float k1 = 0.000000f, k2 = 0.0f;
//  bool configured = false;
//
//  if (!face_cascade.load("C:\\Users\\pica\\Documents\\GitHub\\super-duper-system\\haarcascade_frontalface_default.xml")) {
//    cout << "--(!)Error loading face cascade\n";
//    return -1;
//  }
//
//  if (!capture.isOpened()) {
//    return -1;
//  }
//
//  //Imagenes asociadas a las ventanas
//  Mat init_window_image = Mat(300, 600, CV_8UC3), filter_window_image = Mat(300, 600, CV_8UC3);
//  //Crear ventana inicial
//  namedWindow(NOMBRE_VENTANA_INICIAL); cvui::init(NOMBRE_VENTANA_INICIAL);
//
//  while (1) {
//
//    switch (option) {
//    case -1:
//      cvui::context(NOMBRE_VENTANA_INICIAL);  //indica que los siguientes elementos son de la ventana inicial
//
//      init_window_image = Scalar(10, 10, 10); // Colorear fondo
//      cvui::text(init_window_image, 50, 30, "Elija un filtro", 2 * cvui::DEFAULT_FONT_SCALE);
//
//      if (cvui::button(init_window_image, 50, 60, "Contraste", cvui::DEFAULT_FONT_SCALE, 0x303030)) {
//        option = CONTRAST;
//        openWindow(NOMBRE_VENTANA_CONTRASTE);
//      }
//      if (cvui::button(init_window_image, 50, 90, "Alien", cvui::DEFAULT_FONT_SCALE, 0x303030)) {
//        option = ALIEN;
//        openWindow(NOMBRE_VENTANA_ALIEN);
//      }
//      if (cvui::button(init_window_image, 50, 120, "Poster", cvui::DEFAULT_FONT_SCALE, 0x303030)) {
//        option = POSTER;
//        openWindow(NOMBRE_VENTANA_POSTER);
//      }
//      if (cvui::button(init_window_image, 50, 150, "Distorsion", cvui::DEFAULT_FONT_SCALE, 0x303030)) {
//        option = DISTORTION;
//        openWindow(NOMBRE_VENTANA_DISTORTION);
//      }
//      if (cvui::button(init_window_image, 50, 180, "Emborronado de cara", cvui::DEFAULT_FONT_SCALE, 0x303030)) {
//        option = FACE_BLUR;
//        openWindow(NOMBRE_VENTANA_BLUR);
//      }
//      if (cvui::button(init_window_image, 50, 210, "Fractal", cvui::DEFAULT_FONT_SCALE, 0x303030)) {
//        option = FRACTAL_TRACE;
//        openWindow(NOMBRE_VENTANA_FRACTAL);
//      }
//      if (cvui::button(init_window_image, 50, 240, "Salir", cvui::DEFAULT_FONT_SCALE, 0x303030)) {
//        return 0;
//      }
//      cvui::update(NOMBRE_VENTANA_INICIAL);
//      imshow(NOMBRE_VENTANA_INICIAL, init_window_image);
//
//      break;
//    case CONTRAST:
//      cvui::context(NOMBRE_VENTANA_CONTRASTE);
//
//      filter_window_image = Scalar(10, 10, 10);
//      cvui::text(filter_window_image, 50, 30, "Nivel de contraste", cvui::DEFAULT_FONT_SCALE);
//      cvui::trackbar(filter_window_image, 80, 60, 150, &alpha, 1.0, 3.0, 0.1);
//      if (cvui::button(filter_window_image, 50, 110, "Atras", cvui::DEFAULT_FONT_SCALE, 0x303030)) {
//        option = -1;
//      }
//      if (option == CONTRAST) {
//        cvui::update(NOMBRE_VENTANA_CONTRASTE);
//        imshow(NOMBRE_VENTANA_CONTRASTE, filter_window_image);
//
//        capture.read(image);
//        if (image.empty()) {
//          cout << "Could not load the image from the camera!\n" << endl;
//          return -1;
//        }
//        imshow("Original Image", image);
//        contrast_effect(&image, alpha);
//        imshow("Changed Image", image);
//
//      }
//      else {
//        closeWindow(NOMBRE_VENTANA_CONTRASTE);
//        closeWindow("Original Image");
//        closeWindow("Changed Image");
//      }
//      break;
//    case ALIEN:
//      cvui::context(NOMBRE_VENTANA_ALIEN);
//
//      filter_window_image = Scalar(10, 10, 10);
//      cvui::text(filter_window_image, 50, 30, "Nivel de azul", cvui::DEFAULT_FONT_SCALE);
//      cvui::trackbar(filter_window_image, 50, 60, 150, &blue, 0.0, 1.0, 0.1);
//      cvui::text(filter_window_image, 50, 90, "Nivel de verde", cvui::DEFAULT_FONT_SCALE);
//      cvui::trackbar(filter_window_image, 50, 120, 150, &green, 0.0, 1.0, 0.1);
//      cvui::text(filter_window_image, 50, 150, "Nivel de rojo", cvui::DEFAULT_FONT_SCALE);
//      cvui::trackbar(filter_window_image, 50, 180, 150, &red, 0.0, 1.0, 0.1);
//      if (cvui::button(filter_window_image, 50, 230, "Atras", cvui::DEFAULT_FONT_SCALE, 0x303030)) {
//        option = -1;
//      }
//      if (option == ALIEN) {
//        cvui::update(NOMBRE_VENTANA_ALIEN);
//        imshow(NOMBRE_VENTANA_ALIEN, filter_window_image);
//
//        capture.read(image);
//        if (image.empty()) {
//          cout << "Could not load the image from the camera!\n" << endl;
//          return -1;
//        }
//        imshow("Original Image", image);
//        alien_effect(&image, blue, green, red);
//        imshow("Changed Image", image);
//
//      }
//      else {
//        closeWindow(NOMBRE_VENTANA_ALIEN);
//        closeWindow("Original Image");
//        closeWindow("Changed Image");
//      }
//      break;
//    case DISTORTION:
//      cvui::context(NOMBRE_VENTANA_DISTORTION);
//
//      filter_window_image = Scalar(10, 10, 10);
//      cvui::text(filter_window_image, 50, 30, "k1", cvui::DEFAULT_FONT_SCALE);
//      cvui::trackbar(filter_window_image, 80, 60, 150, &k1, -0.000009f, 0.000009f, 0.000001f, "%.6Lf");
//      //cvui::text(filter_window_image, 50, 90, "k2", cvui::DEFAULT_FONT_SCALE);
//      //cvui::trackbar(filter_window_image, 80, 120, 150, &k2, -0.000009f, 0.000009f, 0.000001f, "%.6Lf");
//      if (cvui::button(filter_window_image, 50, 220, "Atras", cvui::DEFAULT_FONT_SCALE, 0x303030)) {
//        option = -1;
//      }
//      if (option == DISTORTION) {
//        cvui::update(NOMBRE_VENTANA_DISTORTION);
//        imshow(NOMBRE_VENTANA_DISTORTION, filter_window_image);
//
//        capture.read(image);
//        if (image.empty()) {
//          cout << "Could not load the image from the camera!\n" << endl;
//          return -1;
//        }
//        imshow("Original Image", image);
//        distortion_effect(&image, k1, k2);
//        imshow("Changed Image", image);
//
//      }
//      else {
//        closeWindow(NOMBRE_VENTANA_DISTORTION);
//        closeWindow("Original Image");
//        closeWindow("Changed Image");
//      }
//      break;
//    case POSTER:
//      cvui::context(NOMBRE_VENTANA_POSTER);
//
//      filter_window_image = Scalar(10, 10, 10);
//      cvui::text(filter_window_image, 50, 30, "Numero de colores", cvui::DEFAULT_FONT_SCALE);
//      cvui::trackbar(filter_window_image, 80, 60, 150, &n_colors, 1, 8, 1, "2^%1Lf^3");
//      if (cvui::button(filter_window_image, 50, 110, "Atras", cvui::DEFAULT_FONT_SCALE, 0x303030)) {
//        option = -1;
//      }
//      if (option == POSTER) {
//        cvui::update(NOMBRE_VENTANA_POSTER);
//        imshow(NOMBRE_VENTANA_POSTER, filter_window_image);
//
//        capture.read(image);
//        if (image.empty()) {
//          cout << "Could not load the image from the camera!\n" << endl;
//          return -1;
//        }
//        imshow("Original Image", image);
//        poster_effect(&image, 8 - n_colors);
//        imshow("Changed Image", image);
//
//      }
//      else {
//        closeWindow(NOMBRE_VENTANA_POSTER);
//        closeWindow("Original Image");
//        closeWindow("Changed Image");
//      }
//      break;
//    case FACE_BLUR:
//      cvui::context(NOMBRE_VENTANA_BLUR);
//
//      filter_window_image = Scalar(10, 10, 10);
//      cvui::text(filter_window_image, 50, 30, "Factor de blur", cvui::DEFAULT_FONT_SCALE);
//      cvui::trackbar(filter_window_image, 80, 60, 150, &blur_factor, 1, 20, 1);
//      if (cvui::button(filter_window_image, 50, 110, "Atras", cvui::DEFAULT_FONT_SCALE, 0x303030)) {
//        option = -1;
//      }
//      if (option == FACE_BLUR) {
//        cvui::update(NOMBRE_VENTANA_BLUR);
//        imshow(NOMBRE_VENTANA_BLUR, filter_window_image);
//
//        capture.read(image);
//        if (image.empty()) {
//          cout << "Could not load the image from the camera!\n" << endl;
//          return -1;
//        }
//        imshow("Original Image", image);
//        face_blur_effect(&image, &face_cascade, blur_factor);
//        imshow("Changed Image", image);
//
//      }
//      else {
//        closeWindow(NOMBRE_VENTANA_BLUR);
//        closeWindow("Original Image");
//        closeWindow("Changed Image");
//      }
//      break;
//    case FRACTAL_TRACE:
//      cvui::context(NOMBRE_VENTANA_FRACTAL);
//
//      filter_window_image = Scalar(10, 10, 10);
//      cvui::text(filter_window_image, 50, 50, "profundidad", cvui::DEFAULT_FONT_SCALE);
//      cvui::trackbar(filter_window_image, 80, 60, 150, &aux1, 1, 10, 1);
//      cvui::text(filter_window_image, 50, 110, "X", cvui::DEFAULT_FONT_SCALE);
//      cvui::trackbar(filter_window_image, 80, 120, 150, &aux2, 1, 10, 1);
//      cvui::text(filter_window_image, 50, 170, "Y", cvui::DEFAULT_FONT_SCALE);
//      cvui::trackbar(filter_window_image, 80, 180, 150, &aux3, 1, 10, 1);
//      if (cvui::button(filter_window_image, 50, 220, "Atras", cvui::DEFAULT_FONT_SCALE, 0x303030)) {
//        option = -1;
//      }
//      if (option == FRACTAL_TRACE) {
//        cvui::update(NOMBRE_VENTANA_FRACTAL);
//        imshow(NOMBRE_VENTANA_FRACTAL, filter_window_image);
//
//        capture.read(image);
//        if (image.empty()) {
//          cout << "Could not load the image from the camera!\n" << endl;
//          return -1;
//        }
//        imshow("Original Image", image);
//        depth = aux1;
//        factorX = aux2 / 1000.f;
//        factorY = aux3 / 1000.f;
//        fractal_trace_effect(&image, depth, factorX, factorY);
//        imshow("Changed Image", image);
//
//      }
//      else {
//        closeWindow(NOMBRE_VENTANA_FRACTAL);
//        closeWindow("Original Image");
//        closeWindow("Changed Image");
//      }
//      break;
//    case EXIT:
//      return 0;
//    default:
//      cout << "Not a valid option, try again" << endl;
//    }
//
//
//
//    if (waitKey(20) == 27) {
//      break;
//    }
//
//  }
//
//  return 0;
//}
