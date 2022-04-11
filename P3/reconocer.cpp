#include "opencv2/imgcodecs.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include <vector>
//#include <cmath>

using std::to_string;
using std::string;
using std::vector;
using std::cin;
using std::cout;
using std::endl;
using namespace cv;

const string PATH = "C:\\Users\\pica\\Documents\\GitHub\\super-duper-system\\P3\\images\\";
const string DATA_NAME = "objetos.txt";
const int BUFF_LENGTH = 1024;
const int NUM_DESCRIPTORS = 5;
const int NUM_FIELDS = NUM_DESCRIPTORS * 2 + 2;
const int MIN_LENGTH_LINE = NUM_FIELDS;

Mat adapt_gauss_threshold(Mat image, int blur_size, int threshold_type, int block_size, 
    double c);
Mat adapt_mean_threshold(Mat image, int blur_size, int threshold_type, int block_size,
    double c);
Mat otsu_threshold(Mat image, int gauss_size);

//Formato del fichero de aprendizaje:
//<nombre_de_la_figura>;<n>;<media1>;<varianza1>;<media2>;<varianza2>;<media3>;<varianza3>;<media4>;<varianza4>;<media5>;<varianza5>
//<nombre_de_la_figura2>;...
//...
// Siendo 1) area 2) perimetro 3) momento1 4) momento2 5) momento3
int main(int argc, char** argv){
  if(argc != 2){ // Check parameters
    cout << "Wrong number of parameters, usage:\n";
    cout << "reconocer <nomfich>\n";
    exit(1);
  }

  // Lectura de parametros
  string fich_name = argv[1];

  // Lee datos
  char* buffer = new char[BUFF_LENGTH];
  string buffer_s;
  int readedCount, pos;
  vector<string> lines;
  vector<string> classes;
  vector<vector<double>> data;
  vector<double> aux;

  std::ifstream objects(PATH + DATA_NAME);

  if(objects.is_open()){ // Existen datos anteriores
    do{
      objects.read(buffer, BUFF_LENGTH);
      readedCount = objects.gcount();
      buffer_s = buffer_s + string(buffer).substr(0, readedCount);
    }while (readedCount == BUFF_LENGTH);

    // Los divide en lineas
    pos = buffer_s.find("\n");
    while (pos != string::npos){
      if (pos > MIN_LENGTH_LINE){
        lines.push_back(buffer_s.substr(0, pos));
      }
      buffer_s.erase(0, pos + 1);
      pos = buffer_s.find("\n");
    }
    objects.close();

    // Indexa los nombres de las clases y sus datos
    for (int i = 0; i < lines.size(); i++){
      pos = lines[i].find(";");
      classes.push_back(lines[i].substr(0, pos));
      lines[i].erase(0, pos + 1);
      for (int j = 1; j < NUM_FIELDS; j++){
        pos = lines[i].find(";");
        aux.push_back(stod(lines[i].substr(0, pos)));
        lines[i].erase(0, pos + 1);
      }
      data.push_back(aux);
    }

  }else{ // Primer objeto aprendido
    cout << "No encuentra objetos.txt, no se puede reconocer nada\n";
    exit(1);
  }

  // Lee el archivo a aprender
  Mat image = imread(samples::findFile(fich_name), IMREAD_COLOR);
  if(image.empty()){
    printf("Error opening image: %s\n", fich_name.c_str());
    return EXIT_FAILURE;
  }
  imshow("Image to learn", image);

  Mat otsu = otsu_threshold(image, 5);

  imshow("Image otsurized", otsu);

  Mat canny;
  Canny(otsu, canny, 100, 200);
  vector<vector<Point>> contours;
  findContours(canny, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

  // Saca los valores de los descriptores de todos los objetos detectados
  vector<float> areas;
  vector<float> perims;
  vector<double> firstHuMoments;
  vector<double> secondHuMoments;
  vector<double> thirdHuMoments;
  for (int i = 0; i < contours.size(); i++){
    areas.push_back(contourArea(contours[i]));
    perims.push_back(arcLength(contours[i], true));
    Moments mu = moments(contours[i]);  
    double huMoments[3] = {0.0, 0.0, 0.0};
    for(int moment = 0; moment < 3; moment++){
      huMoments[moment] = -1 * copysign(1.0, huMoments[moment]) 
          * log10(abs(huMoments[moment]));
    }
    firstHuMoments.push_back(huMoments[0]);
    secondHuMoments.push_back(huMoments[1]);
    thirdHuMoments.push_back(huMoments[2]);
  }

  // Covars matrixes
  vector<Mat> covars;
  for (int classes = 0; classes < data.size(); classes++){
    Mat covar = Mat_<double>(5, 5);
    double variances[5];
    int n = data[classes][0];
    variances[0] = data[classes][2] * n;
    variances[1] = data[classes][4] * n;
    variances[2] = data[classes][6] * n;
    variances[3] = data[classes][8] * n;
    variances[4] = data[classes][10] * n;
    for(int i = 0; i < 5; i++){
      for(int j = 0; j < 5; j++){
        covar.at<double>(i, j) = variances[i] * variances[j] / n;
      }
    }
    covars.push_back(covar);
  }

  // Mahalanobis
  double array1;
  double array2;
  Mat matrix;
  vector<double> distances;
  for (int contourClass = 0; contourClass < data.size(); contourClass++){
    for (int contour = 0; contour < contours.size(); contour++){
      distances.push_back(Mahalanobis(array1, array2, matrix));
    }
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
