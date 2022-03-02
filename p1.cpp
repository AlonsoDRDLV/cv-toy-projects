#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "cvui.h"
#include <iostream>
#include <cmath>

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

//Tener en cuenta overflow
void contrast_effect(Mat* image, double alpha){
  for (int y = 0; y < image->rows; y++){
    for (int x = 0; x < image->cols; x++){
      for (int c = 0; c < image->channels(); c++){
        image->at<Vec3b>(y, x)[c] =
          saturate_cast<uchar>(alpha * image->at<Vec3b>(y, x)[c]);
      }
    }
  }
}

void alien_effect(Mat* image, double blue, double green, double red){
  Mat image_HSV;
  Mat image_YCrCb;
  cvtColor(*image, image_HSV, CV_BGR2HSV);
  cvtColor(*image, image_YCrCb, CV_BGR2YCrCb);
  double b, g, r, h, s, y, cr, cb;
  for (int row = 0; row < image->rows; row++){
    for (int col = 0; col < image->cols; col++){
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
          (cr <= (-2.2857 * cb) + 432.85))){
        image->at<Vec3b>(row, col)[0] = b * blue;
        image->at<Vec3b>(row, col)[1] = g * green;
        image->at<Vec3b>(row, col)[2] = r * red;
      }
    }
  }
}

//reduce to 2^div colors
void poster_effect(Mat* img, int div = 5){
  uchar mask = 0xFF;
  mask <<= div;
  for (int i = 0; i < img->rows; i++){
    uchar* data = img->ptr<uchar>(i);
    for (int j = 0; j < img->cols * img->channels(); j++){
      data[j] &= mask;
    }
  }
}

void distortion_effect(Mat* image, float k1, float k2){
  //Mat aux = *image;
  Mat aux(image->size(), 16);
  int yPrev, xPrev, xCenter, yCenter;
  double rSquare, disX, disY;
  yCenter = aux.rows / 2;
  xCenter = aux.cols / 2;
  for (int y = 0; y < image->rows; y++){
    for (int x = 0; x < image->cols; x++){
      disY = y - yCenter;
      disX = x - xCenter;
      rSquare = pow(disX, 2) + pow(disY, 2);
      yPrev = y + disY * k1 * rSquare + disY * k2 * pow(rSquare, 2);
      xPrev = x + disX * k1 * rSquare + disX * k2 * pow(rSquare, 2);
      if ((yPrev < aux.rows) && (xPrev < aux.cols) && (yPrev >= 0) && (xPrev >= 0)){
        aux.at<Vec3b>(y, x) = image->at<Vec3b>(yPrev, xPrev);
      }
    }
  }
  *image = aux;
}

void face_blur_effect(Mat* image, CascadeClassifier* face_cascade, int blur_factor){
  Mat gray_image(image->size(), 16);
  cvtColor(*image, gray_image, COLOR_BGR2GRAY);
  std::vector<Rect> faces;
  face_cascade->detectMultiScale(gray_image, faces); // solo acepta grises el classifier por eso la imagen gris
  int x, y, h, w;

  if (blur_factor >= 1){
    for (int i = 0; i < faces.size(); i++){
      x = faces[i].x;
      y = faces[i].y;
      w = faces[i].width;
      h = faces[i].height;
      Mat face_mat(h, w, 16);
      Rect face_rect(x, y, w, h);
      Mat aux = *image; // Quiero quitar esta línea pero no sé cómo
      aux(face_rect).copyTo(face_mat);
      medianBlur(face_mat, face_mat, blur_factor * 2 + 1); // la fórmula rara es porque solo acepta impares
      face_mat.copyTo(aux(face_rect));
    }
  }
}

void fractal_trace_effect(Mat* image, int depth, float factX, float factY){
  Mat aux(image->size(), 16);
  double auxX, auxY, newX, newY, tmp;

  for (int y = 0; y < image->rows; y++){
    for (int x = 0; x < image->cols; x++){
      auxX = x;
      auxY = y;
      for (int i = 0; i < depth; i++){
        newX = auxX * auxX;
        newY = auxY * auxY;
        auxY = int(factY * (2 * auxX * auxY + auxY)) % aux.rows;
        auxX = int(factX * (newX - newY + auxX)) % aux.cols;
        if (auxX < 0){
          auxX += aux.cols;
        }
        if (auxY < 0){
          auxY += aux.rows;
        }
      }
      aux.at<Vec3b>(y, x) = image->at<Vec3b>(auxY, auxX);
    }
  }
  *image = aux;
}

int main(int argc, char** argv){
  VideoCapture capture;
  capture.open(0);
  CascadeClassifier face_cascade;
  Mat image;
  int alpha = 1; /*< Simple contrast control */
  int option, div, n_colors, blur_factor, depth;
  double blue, green, red;
  float factorX, factorY;
  int aux1, aux2, aux3;
  aux1 = aux2 = aux3 = 1;
  float k1, k2;
  bool configured = false;

  if (!face_cascade.load("C:\\Users\\pica\\Documents\\GitHub\\super-duper-system\\haarcascade_frontalface_default.xml")){
    cout << "--(!)Error loading face cascade\n";
    return -1;
  }

  if (!capture.isOpened()){
    return -1;
  }

  while (1){
    cout << " Basic Linear Transforms " << endl;
    cout << "-------------------------" << endl;
    cout << "* Enter the filter to apply: " << endl;
    cout << "* " << CONTRAST << ") Contrast effect" << endl;
    cout << "* " << ALIEN << ") Alien effect" << endl;
    cout << "* " << DISTORTION << ") Distortion effect" << endl;
    cout << "* " << POSTER << ") Poster effect" << endl;
    cout << "* " << FACE_BLUR << ") Face blur effect" << endl;
    cout << "* " << FRACTAL_TRACE << ") Fractal trace effect" << endl;
    cout << "* " << EXIT << ") Exit" << endl;
    cin >> option;

    switch (option){
    case CONTRAST:
      //cout << "* Enter the alpha value [1.0-3.0]: ";
      //cin >> alpha;
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
    case FACE_BLUR:
      cout << "* blur factor please:";
      cin >> blur_factor;
      break;
    case FRACTAL_TRACE:
      cout << "* depth please (an integer ej:3):";
      cin >> depth;
      cout << "* horizontal factor please (an integer ej:3):";
      cin >> factorX;
      cout << "* vertical factor please (an integer ej:3):";
      cin >> factorY;
      factorX = factorX/ 1000;
      factorY = factorY/ 1000;
      break;
    case EXIT:
      return 0;
    default:
      cout << "Not a valid option, try again" << endl;
    }


    while (true){
      capture.read(image);
      if (image.empty()){
        cout << "Could not load the image from the camera!\n" << endl;
        return -1;
      }


      imshow("Original Image", image);

      switch (option){
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
      case FACE_BLUR:
        face_blur_effect(&image, &face_cascade, blur_factor);
        break;
      case FRACTAL_TRACE:
        createTrackbar("coso1", "Original Image", &aux1, 100);
        createTrackbar("coso2", "Original Image", &aux2, 100);
        createTrackbar("coso3", "Original Image", &aux3, 100);
        depth = aux1;
        factorX = aux2 / 1000.f;
        factorY = aux3 / 1000.f;
        fractal_trace_effect(&image, depth, factorX, factorY);
        break;
      }
      imshow("Changed Image", image);


      int c = waitKey(10);
      if ((char)c == 'q'){
        cvDestroyWindow("Original Image");
        break;
      }
    }
  }

  return 0;
}
