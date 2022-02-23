#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "cvui.h"
#include <iostream>
#include <cmath>

using std::cin;
using std::cout;
using std::endl;
using namespace cv;

const int CONTRAST = 0;
const int ALIEN = 1;
const int POSTER = 3;
const int DISTORTION = 2;
const int EXIT = 4;


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

void distortion_effect(Mat* image, double k1, double k2) {
  Mat aux = *image;
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
        aux.at<Vec3b>(yPrev, xPrev) = image->at<Vec3b>(y, x);
      }
      if (rSquare < 30) {
        cout << rSquare << " " << yPrev << " " << xPrev << " " << y << " " << x << endl;

      }
    }
  }
  *image = aux;
}

int main(int argc, char** argv) {
  VideoCapture capture;
  capture.open(0);
  Mat image;
  double alpha = 1.0; /*< Simple contrast control */
  int option, div, n_colors;
  double blue, green, red, k1, k2;
  bool configured = false;

  if (!capture.isOpened()) {
    return -1;
  }

  while (1) {
    cout << " Basic Linear Transforms " << endl;
    cout << "-------------------------" << endl;
    cout << "* Enter the filter to apply: " << endl;
    cout << "* " << CONTRAST << ") Contrast effect" << endl;
    cout << "* " << ALIEN << ") Alien effect" << endl;
    cout << "* " << DISTORTION << ") Distortion effect" << endl;
    cout << "* " << POSTER << ") Poster effect" << endl;
    cout << "* " << EXIT << ") Exit" << endl;
    cin >> option;

    switch (option) {
    case CONTRAST:
      cout << "* Enter the alpha value [1.0-3.0]: ";
      cin >> alpha;
      break;
    case ALIEN:
      cout << "* Enter the blue value reduce factor [0.0-1.0]: ";
      cin >> blue;
      cout << "* Enter the green value reduce factor [0.0-1.0]: ";
      cin >> green;
      cout << "* Enter the red value reduce factor [0.0-1.0]: ";
      cin >> red;
      break;
    case DISTORTION:
      cout << "* Enter the k1 value: ";
      cin >> k1;
      cout << "* Enter the k2 value: ";
      cin >> k2;
      break;
    case POSTER:
      cout << "* Enter the number of colors of the new palete (a power of 2 if you want to see the exact number of colors you've said): ";
      cin >> n_colors;
      div = 8 - log2(n_colors);
      break;
    case EXIT:
      return 0;
    default:
      cout << "Not a valid option, try again" << endl;
    }


    while (true) {
      capture.read(image);
      if (image.empty()) {
        cout << "Could not load the image from the camera!\n" << endl;
        return -1;
      }

      switch (option) {
      case CONTRAST:
        contrast_effect(&image, alpha);
        break;
      case ALIEN:
        alien_effect(&image, blue, green, red);
        break;
      case DISTORTION:
        distortion_effect(&image, k1, k2);
        break;
      case POSTER:
        poster_effect(&image, div);
        break;
      }

      imshow("Original Image", image);

      int c = waitKey(10);
      if ((char)c == 'q') {
        cvDestroyWindow("Original Image");
        break;
      }
    }
  }

  return 0;
}
