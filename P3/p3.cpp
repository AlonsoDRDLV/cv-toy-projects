#include "opencv2/imgcodecs.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <string>
#include <iomanip>
//#include <cmath>

#define CVUI_IMPLEMENTATION
#include "cvui.h"

using std::to_string;
using std::string;
using std::vector;
using std::cin;
using std::cout;
using std::endl;
using namespace cv;

Mat adapt_gauss_threshold(Mat image, int blur_size, int threshold_type, int block_size, 
    double c);
Mat adapt_mean_threshold(Mat image, int blur_size, int threshold_type, int block_size,
    double c);
Mat otsu_threshold(Mat image, int gauss_size);

int main(int argc, char** argv){
  Mat image, copy, altered_image, opencv_image;
  string imagePath = "C:\\Users\\pica\\Documents\\GitHub\\super-duper-system\\P3\\images\\";
  string imageName = imagePath + "reco1.pgm";

  image = imread(samples::findFile(imageName), IMREAD_COLOR);
  if(image.empty()){
    printf("Error opening image: %s\n", imageName.c_str());
    return EXIT_FAILURE;
  }
  imshow("coso", image);

  // Dice que lo que valoran en esta práctica es que hayamos descubierto todos los caminos para
  // resolver todo esto, imagino que vamos bien, al menos al principio, mi materia gris ya chof hoy

  // Esta parte de los threshold está muy tocada para más o menos lo que creo que es, pero hay que
  // interpretar muy fuerte el enunciado para saber qué diablos quieren. . .
  // Aumentar block_size hace los contornos más finos
  // ¿Aumentar c reduce el ruido?
  // Gaussiano parece mejor por ahora
  // Como interesa el objeto igual es interesante coger grosor ancho y rellenar (teacher words)
  Mat adapt_mean = 
      adapt_mean_threshold(image, 7, THRESH_BINARY, 71, 2);
  imshow("adapt mean threshold", adapt_mean);
  Mat adapt_gauss =
      adapt_gauss_threshold(image, 7, THRESH_BINARY, 131, 2);
  imshow("adapt gauss threshold", adapt_gauss);

  // Otsu, casi la solución perfecta, puesto que deja un puntito y creo
  // que también tapa el hueco de la sierra y no debería hacerlo, si se aumenta
  // el size del filtro gaussiano se queda tal cual las figuras todo sombreadas
  // pero lo dicho: igual no es lo suyo puesto que confundirá igual círculos
  Mat otsu = otsu_threshold(image, 5);
  imshow("otsu threshold", otsu);

  // Saca la línea azul esa fea a la izquierda, tengo que ver cómo retirarla
  Mat labelImage(otsu.size(), CV_32S);
  int nLabels = connectedComponents(otsu, labelImage, 8);
  vector<Vec3b> colors(nLabels);
  colors[0] = Vec3b(0, 0, 0);//background
  
  for (int label = 1; label < nLabels; ++label){
    colors[label] = Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
  }

  Mat* comp = new Mat[nLabels];
  for (int i = 1; i < nLabels; i++){
    comp[i - 1] = Mat(otsu.size(), CV_8UC1, Scalar(0));
    for (int r = 0; r < otsu.rows; ++r){
      for (int c = 0; c < otsu.cols; ++c){
        int pixel = labelImage.at<int>(r, c);
        if (pixel == i){
          comp[i - 1].at<uchar>(r, c) = 255;
        }else{
          comp[i - 1].at<uchar>(r, c) = 0;
        }
      }
    }
    imshow("Connected Components " + to_string(i), comp[i - 1]);
  }

  Mat canny;
  Canny(otsu, canny, 100, 200);
  vector<vector<Point>> contours;
  findContours(canny, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

  // live the past, forget the moment, o algo así
  vector<Moments> mu(contours.size());
  for(size_t i = 0; i < contours.size(); i++){
    mu[i] = moments(contours[i]);
  }

  // Me ha costado pero esto es los moment center
  vector<Point2f> mc(contours.size());
  for(size_t i = 0; i < contours.size(); i++){
    //add 1e-5 to avoid division by zero
    mc[i] = Point2f(static_cast<float>(mu[i].m10 / (mu[i].m00 + 1e-5)),
        static_cast<float>(mu[i].m01 / (mu[i].m00 + 1e-5)));
    cout << "mc[" << i << "]=" << mc[i] << endl;
  }
  
  // las dibujaciones
  Mat drawing = Mat::zeros(canny.size(), CV_8UC3);
  for(size_t i = 0; i < contours.size(); i++){
    Scalar color = Scalar(rand() & 255, rand() & 255, rand() & 255);
    drawContours(drawing, contours, (int)i, color, 2);
    circle(drawing, mc[i], 4, color, -1);
  }

  // las imprimiciones
  imshow("Contours", drawing);
  cout << "\t Info: Area and Contour Length \n";
  for(size_t i = 0; i < contours.size(); i++){
    cout << " * Contour[" << i << "] - Area (M_00) = " << std::fixed << std::setprecision(2) << mu[i].m00
        << " - Area OpenCV: " << contourArea(contours[i]) << " - Length: " << arcLength(contours[i], true) << endl;
  }

  waitKey(0);

  return EXIT_SUCCESS;
}

Mat adapt_gauss_threshold(Mat image, int blur_size, int threshold_type, int block_size, 
    double c){
  Mat result = image.clone();
  cvtColor(result, result, COLOR_BGR2GRAY);
  medianBlur(result, result, 5);
  adaptiveThreshold(result, result, 255, ADAPTIVE_THRESH_GAUSSIAN_C, threshold_type, 
      block_size, c);
  result = Mat(result.size(), result.type(), 255) - result;
  return result;
}

Mat adapt_mean_threshold(Mat image, int blur_size, int threshold_type, int block_size,
  double c){
  Mat result = image.clone();
  cvtColor(result, result, COLOR_BGR2GRAY);
  medianBlur(result, result, 5);
  adaptiveThreshold(result, result, 255, ADAPTIVE_THRESH_MEAN_C, threshold_type,
      block_size, c);
  result = Mat(result.size(), result.type(), 255) - result;
  return result;
}

Mat otsu_threshold(Mat image, int gauss_size){
  Mat result = image.clone();
  cvtColor(result, result, COLOR_BGR2GRAY);
  GaussianBlur(result, result, Size(gauss_size, gauss_size), 0, 0);
  threshold(result, result, 0, 255, THRESH_BINARY + THRESH_OTSU);
  result = Mat(result.size(), result.type(), 255) - result;
  return result;
}
