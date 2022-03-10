#include "opencv2/imgcodecs.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <cmath>

#define CVUI_IMPLEMENTATION
#include "cvui.h"

using std::cin;
using std::cout;
using std::endl;
using namespace cv;

const int CONTRAST = 0;
const int ALIEN = 1;
const int DISTORTION = 2;
const int POSTER = 3;
const int FACE_BLUR = 4;
const int FRACTAL_TRACE = 5;
const int EXIT = 6;

const int ALPHA_MAX = 3;

#define NOMBRE_VENTANA_INICIAL "VC Lab1"
#define NOMBRE_VENTANA_CONTRASTE "VC Lab1 - Contraste"
#define NOMBRE_VENTANA_ALIEN "VC Lab1 - Alien"
#define NOMBRE_VENTANA_POSTER "VC Lab1 - Poster"
#define NOMBRE_VENTANA_BLUR "VC Lab1 - Blur"
#define NOMBRE_VENTANA_FRACTAL "VC Lab1 - Fractal"
#define NOMBRE_VENTANA_DISTORTION "VC Lab1 - Distorsion"

//Tener en cuenta overflow
void contrast_effect(Mat* image, double alpha) {
  for (int y = 0; y < image->rows; y++) {
    for (int x = 0; x < image->cols; x++) {
      for (int c = 0; c < image->channels(); c++) {
        image->at<Vec3b>(y, x)[c] =
          saturate_cast<uchar>(alpha * image->at<Vec3b>(y, x)[c]);
      }
    }
  }
}

void alien_effect(Mat* image, double blue, double green, double red) {
  Mat image_HSV;
  Mat image_YCrCb;
  cvtColor(*image, image_HSV, CV_BGR2HSV);
  cvtColor(*image, image_YCrCb, CV_BGR2YCrCb);
  double b, g, r, h, s, y, cr, cb;
  for (int row = 0; row < image->rows; row++) {
    for (int col = 0; col < image->cols; col++) {
      b = image->at<Vec3b>(row, col)[0];
      g = image->at<Vec3b>(row, col)[1];
      r = image->at<Vec3b>(row, col)[2];
      h = image_HSV.at<Vec3b>(row, col)[0];
      s = image_HSV.at<Vec3b>(row, col)[1];
      y = image_YCrCb.at<Vec3b>(row, col)[0];
      cr = image_YCrCb.at<Vec3b>(row, col)[1];
      cb = image_YCrCb.at<Vec3b>(row, col)[2];
      if (((0 <= h) && (h <= 50) && (0.23 <= s) && (s <= 0.68) &&
        (r > 95) && (g > 40) && (b > 20) && (r > g) && (r > b)
        && (abs(r - g) > 15))
        ||
        ((r > 95) && (g > 40) && (b > 20) && (r > b)
          && (abs(r - g) > 15) && (cr > 135) &&
          (cb > 85) && (y > 80) && (cr <= (1.5862 * cb) + 20) &&
          (cr >= (0.3448 * cb) + 76.2069) &&
          (cr >= (-4.5652 * cb) + 234.5652) &&
          (cr <= (-1.15 * cb) + 301.75) &&
          (cr <= (-2.2857 * cb) + 432.85))) {
        image->at<Vec3b>(row, col)[0] = b * blue;
        image->at<Vec3b>(row, col)[1] = g * green;
        image->at<Vec3b>(row, col)[2] = r * red;
      }
    }
  }
}

//reduce to 2^div colors
void poster_effect(Mat* img, int div = 5) {
  uchar mask = 0xFF;
  mask <<= div;
  for (int i = 0; i < img->rows; i++) {
    uchar* data = img->ptr<uchar>(i);
    for (int j = 0; j < img->cols * img->channels(); j++) {
      data[j] &= mask;
    }
  }
}

void distortion_effect(Mat* image, float k1, float k2) {
  //Mat aux = *image;
  Mat aux(image->size(), 16);
  int yPrev, xPrev, xCenter, yCenter;
  double rSquare, disX, disY;
  yCenter = aux.rows / 2;
  xCenter = aux.cols / 2;
  for (int y = 0; y < image->rows; y++) {
    for (int x = 0; x < image->cols; x++) {
      disY = y - yCenter;
      disX = x - xCenter;
      rSquare = pow(disX, 2) + pow(disY, 2);
      yPrev = y + disY * k1 * rSquare + disY * k2 * pow(rSquare, 2);
      xPrev = x + disX * k1 * rSquare + disX * k2 * pow(rSquare, 2);
      if ((yPrev < aux.rows) && (xPrev < aux.cols) && (yPrev >= 0) && (xPrev >= 0)) {
        aux.at<Vec3b>(y, x) = image->at<Vec3b>(yPrev, xPrev);
      }
    }
  }
  *image = aux;
}

void face_blur_effect(Mat* image, CascadeClassifier* face_cascade, int blur_factor) {
  Mat gray_image(image->size(), 16);
  cvtColor(*image, gray_image, COLOR_BGR2GRAY);
  std::vector<Rect> faces;
  face_cascade->detectMultiScale(gray_image, faces); // solo acepta grises el classifier por eso la imagen gris
  int x, y, h, w;

  if (blur_factor >= 1) {
    for (int i = 0; i < faces.size(); i++) {
      x = faces[i].x;
      y = faces[i].y;
      w = faces[i].width;
      h = faces[i].height;
      Mat face_mat(h, w, 16);
      Rect face_rect(x, y, w, h);
      Mat aux = *image; // Quiero quitar esta l�nea pero no s� c�mo
      aux(face_rect).copyTo(face_mat);
      medianBlur(face_mat, face_mat, blur_factor * 2 + 1); // la f�rmula rara es porque solo acepta impares
      face_mat.copyTo(aux(face_rect));
    }
  }
}

void fractal_trace_effect(Mat* image, int depth, float factX, float factY) {
  Mat aux(image->size(), 16);
  double auxX, auxY, newX, newY, tmp;

  for (int y = 0; y < image->rows; y++) {
    for (int x = 0; x < image->cols; x++) {
      auxX = x;
      auxY = y;
      for (int i = 0; i < depth; i++) {
        newX = auxX * auxX;
        newY = auxY * auxY;
        auxY = int(factY * (2 * auxX * auxY + auxY)) % aux.rows;
        auxX = int(factX * (newX - newY + auxX)) % aux.cols;
        if (auxX < 0) {
          auxX += aux.cols;
        }
        if (auxY < 0) {
          auxY += aux.rows;
        }
      }
      aux.at<Vec3b>(y, x) = image->at<Vec3b>(auxY, auxX);
    }
  }
  *image = aux;
}

// From: https://stackoverflow.com/a/48055987/29827
int isWindowOpen(const cv::String& name) {
  return cv::getWindowProperty(name, cv::WND_PROP_AUTOSIZE) != -1;
}

// Open a new OpenCV window and watch it using cvui
void openWindow(const cv::String& name) {
  cv::namedWindow(name);
  cvui::watch(name);
}

// Open an OpenCV window
void closeWindow(const cv::String& name) {
  cv::destroyWindow(name);

  // Ensure OpenCV window event queue is processed, otherwise the window
  // will not be closed.
  cv::waitKey(1);
}

int main(int argc, char** argv) {
  VideoCapture capture;
  capture.open(0);
  CascadeClassifier face_cascade;
  Mat image;
  double alpha = 1; /*< Simple contrast control */
  int option = -1, div, n_colors = 3, blur_factor = 5, depth;
  double blue = 0.0, green = 0.0, red = 0.0;
  float factorX, factorY;
  int aux1, aux2, aux3;
  aux1 = aux2 = aux3 = 1;
  float k1 = 0.000000f, k2 = 0.0f;
  bool configured = false;

  if (!face_cascade.load("C:\\Users\\pica\\Documents\\GitHub\\super-duper-system\\haarcascade_frontalface_default.xml")) {
    cout << "--(!)Error loading face cascade\n";
    return -1;
  }

  if (!capture.isOpened()) {
    return -1;
  }

  //Imagenes asociadas a las ventanas
  Mat init_window_image = Mat(300, 600, CV_8UC3), filter_window_image = Mat(300, 600, CV_8UC3);
  //Crear ventana inicial
  namedWindow(NOMBRE_VENTANA_INICIAL); cvui::init(NOMBRE_VENTANA_INICIAL);

  while (1) {

    switch (option) {
    case -1:
      cvui::context(NOMBRE_VENTANA_INICIAL);  //indica que los siguientes elementos son de la ventana inicial

      init_window_image = Scalar(10, 10, 10); // Colorear fondo
      cvui::text(init_window_image, 50, 30, "Elija un filtro", 2 * cvui::DEFAULT_FONT_SCALE);

      if (cvui::button(init_window_image, 50, 60, "Contraste", cvui::DEFAULT_FONT_SCALE, 0x303030)) {
        option = CONTRAST;
        openWindow(NOMBRE_VENTANA_CONTRASTE);
      }
      if (cvui::button(init_window_image, 50, 90, "Alien", cvui::DEFAULT_FONT_SCALE, 0x303030)) {
        option = ALIEN;
        openWindow(NOMBRE_VENTANA_ALIEN);
      }
      if (cvui::button(init_window_image, 50, 120, "Poster", cvui::DEFAULT_FONT_SCALE, 0x303030)) {
        option = POSTER;
        openWindow(NOMBRE_VENTANA_POSTER);
      }
      if (cvui::button(init_window_image, 50, 150, "Distorsion", cvui::DEFAULT_FONT_SCALE, 0x303030)) {
        option = DISTORTION;
        openWindow(NOMBRE_VENTANA_DISTORTION);
      }
      if (cvui::button(init_window_image, 50, 180, "Emborronado de cara", cvui::DEFAULT_FONT_SCALE, 0x303030)) {
        option = FACE_BLUR;
        openWindow(NOMBRE_VENTANA_BLUR);
      }
      if (cvui::button(init_window_image, 50, 210, "Fractal", cvui::DEFAULT_FONT_SCALE, 0x303030)) {
        option = FRACTAL_TRACE;
        openWindow(NOMBRE_VENTANA_FRACTAL);
      }
      if (cvui::button(init_window_image, 50, 240, "Salir", cvui::DEFAULT_FONT_SCALE, 0x303030)) {
        return 0;
      }
      cvui::update(NOMBRE_VENTANA_INICIAL);
      imshow(NOMBRE_VENTANA_INICIAL, init_window_image);

      break;
    case CONTRAST:
      cvui::context(NOMBRE_VENTANA_CONTRASTE);

      filter_window_image = Scalar(10, 10, 10);
      cvui::text(filter_window_image, 50, 30, "Nivel de contraste", cvui::DEFAULT_FONT_SCALE);
      cvui::trackbar(filter_window_image, 80, 60, 150, &alpha, 1.0, 3.0, 0.1);
      if (cvui::button(filter_window_image, 50, 110, "Atras", cvui::DEFAULT_FONT_SCALE, 0x303030)) {
        option = -1;
      }
      if (option == CONTRAST) {
        cvui::update(NOMBRE_VENTANA_CONTRASTE);
        imshow(NOMBRE_VENTANA_CONTRASTE, filter_window_image);

        capture.read(image);
        if (image.empty()) {
          cout << "Could not load the image from the camera!\n" << endl;
          return -1;
        }
        imshow("Original Image", image);
        contrast_effect(&image, alpha);
        imshow("Changed Image", image);

      }
      else {
        closeWindow(NOMBRE_VENTANA_CONTRASTE);
        closeWindow("Original Image");
        closeWindow("Changed Image");
      }
      break;
    case ALIEN:
      cvui::context(NOMBRE_VENTANA_ALIEN);

      filter_window_image = Scalar(10, 10, 10);
      cvui::text(filter_window_image, 50, 30, "Nivel de azul", cvui::DEFAULT_FONT_SCALE);
      cvui::trackbar(filter_window_image, 50, 60, 150, &blue, 0.0, 1.0, 0.1);
      cvui::text(filter_window_image, 50, 90, "Nivel de verde", cvui::DEFAULT_FONT_SCALE);
      cvui::trackbar(filter_window_image, 50, 120, 150, &green, 0.0, 1.0, 0.1);
      cvui::text(filter_window_image, 50, 150, "Nivel de rojo", cvui::DEFAULT_FONT_SCALE);
      cvui::trackbar(filter_window_image, 50, 180, 150, &red, 0.0, 1.0, 0.1);
      if (cvui::button(filter_window_image, 50, 230, "Atras", cvui::DEFAULT_FONT_SCALE, 0x303030)) {
        option = -1;
      }
      if (option == ALIEN) {
        cvui::update(NOMBRE_VENTANA_ALIEN);
        imshow(NOMBRE_VENTANA_ALIEN, filter_window_image);

        capture.read(image);
        if (image.empty()) {
          cout << "Could not load the image from the camera!\n" << endl;
          return -1;
        }
        imshow("Original Image", image);
        alien_effect(&image, blue, green, red);
        imshow("Changed Image", image);

      }
      else {
        closeWindow(NOMBRE_VENTANA_ALIEN);
        closeWindow("Original Image");
        closeWindow("Changed Image");
      }
      break;
    case DISTORTION:
      cvui::context(NOMBRE_VENTANA_DISTORTION);

      filter_window_image = Scalar(10, 10, 10);
      cvui::text(filter_window_image, 50, 30, "k1", cvui::DEFAULT_FONT_SCALE);
      cvui::trackbar(filter_window_image, 80, 60, 150, &k1, -0.000009f, 0.000009f, 0.000001f, "%.6Lf");
      //cvui::text(filter_window_image, 50, 90, "k2", cvui::DEFAULT_FONT_SCALE);
      //cvui::trackbar(filter_window_image, 80, 120, 150, &k2, -0.000009f, 0.000009f, 0.000001f, "%.6Lf");
      if (cvui::button(filter_window_image, 50, 220, "Atras", cvui::DEFAULT_FONT_SCALE, 0x303030)) {
        option = -1;
      }
      if (option == DISTORTION) {
        cvui::update(NOMBRE_VENTANA_DISTORTION);
        imshow(NOMBRE_VENTANA_DISTORTION, filter_window_image);

        capture.read(image);
        if (image.empty()) {
          cout << "Could not load the image from the camera!\n" << endl;
          return -1;
        }
        imshow("Original Image", image);
        distortion_effect(&image, k1, k2);
        imshow("Changed Image", image);

      }
      else {
        closeWindow(NOMBRE_VENTANA_DISTORTION);
        closeWindow("Original Image");
        closeWindow("Changed Image");
      }
      break;
    case POSTER:
      cvui::context(NOMBRE_VENTANA_POSTER);

      filter_window_image = Scalar(10, 10, 10);
      cvui::text(filter_window_image, 50, 30, "Numero de colores", cvui::DEFAULT_FONT_SCALE);
      cvui::trackbar(filter_window_image, 80, 60, 150, &n_colors, 1, 8, 1, "2^%1Lf^3");
      if (cvui::button(filter_window_image, 50, 110, "Atras", cvui::DEFAULT_FONT_SCALE, 0x303030)) {
        option = -1;
      }
      if (option == POSTER) {
        cvui::update(NOMBRE_VENTANA_POSTER);
        imshow(NOMBRE_VENTANA_POSTER, filter_window_image);

        capture.read(image);
        if (image.empty()) {
          cout << "Could not load the image from the camera!\n" << endl;
          return -1;
        }
        imshow("Original Image", image);
        poster_effect(&image, 8 - n_colors);
        imshow("Changed Image", image);

      }
      else {
        closeWindow(NOMBRE_VENTANA_POSTER);
        closeWindow("Original Image");
        closeWindow("Changed Image");
      }
      break;
    case FACE_BLUR:
      cvui::context(NOMBRE_VENTANA_BLUR);

      filter_window_image = Scalar(10, 10, 10);
      cvui::text(filter_window_image, 50, 30, "Factor de blur", cvui::DEFAULT_FONT_SCALE);
      cvui::trackbar(filter_window_image, 80, 60, 150, &blur_factor, 1, 20, 1);
      if (cvui::button(filter_window_image, 50, 110, "Atras", cvui::DEFAULT_FONT_SCALE, 0x303030)) {
        option = -1;
      }
      if (option == FACE_BLUR) {
        cvui::update(NOMBRE_VENTANA_BLUR);
        imshow(NOMBRE_VENTANA_BLUR, filter_window_image);

        capture.read(image);
        if (image.empty()) {
          cout << "Could not load the image from the camera!\n" << endl;
          return -1;
        }
        imshow("Original Image", image);
        face_blur_effect(&image, &face_cascade, blur_factor);
        imshow("Changed Image", image);

      }
      else {
        closeWindow(NOMBRE_VENTANA_BLUR);
        closeWindow("Original Image");
        closeWindow("Changed Image");
      }
      break;
    case FRACTAL_TRACE:
      cvui::context(NOMBRE_VENTANA_FRACTAL);

      filter_window_image = Scalar(10, 10, 10);
      cvui::text(filter_window_image, 50, 50, "profundidad", cvui::DEFAULT_FONT_SCALE);
      cvui::trackbar(filter_window_image, 80, 60, 150, &aux1, 1, 10, 1);
      cvui::text(filter_window_image, 50, 110, "X", cvui::DEFAULT_FONT_SCALE);
      cvui::trackbar(filter_window_image, 80, 120, 150, &aux2, 1, 10, 1);
      cvui::text(filter_window_image, 50, 170, "Y", cvui::DEFAULT_FONT_SCALE);
      cvui::trackbar(filter_window_image, 80, 180, 150, &aux3, 1, 10, 1);
      if (cvui::button(filter_window_image, 50, 220, "Atras", cvui::DEFAULT_FONT_SCALE, 0x303030)) {
        option = -1;
      }
      if (option == FRACTAL_TRACE) {
        cvui::update(NOMBRE_VENTANA_FRACTAL);
        imshow(NOMBRE_VENTANA_FRACTAL, filter_window_image);

        capture.read(image);
        if (image.empty()) {
          cout << "Could not load the image from the camera!\n" << endl;
          return -1;
        }
        imshow("Original Image", image);
        depth = aux1;
        factorX = aux2 / 1000.f;
        factorY = aux3 / 1000.f;
        fractal_trace_effect(&image, depth, factorX, factorY);
        imshow("Changed Image", image);

      }
      else {
        closeWindow(NOMBRE_VENTANA_FRACTAL);
        closeWindow("Original Image");
        closeWindow("Changed Image");
      }
      break;
    case EXIT:
      return 0;
    default:
      cout << "Not a valid option, try again" << endl;
    }



    if (waitKey(20) == 27) {
      break;
    }

  }

  return 0;
}
