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

const string PATH = "C:\\Users\\AlonsoDRDLV\\Documents\\GitHub\\super-duper-system\\P3\\images\\";
const string DATA_NAME = "objetos.txt";
const double VAR_PERCENTAGE = 0.02;
const int BUFF_LENGTH = 1024;
const int NUM_DESCRIPTORS = 5;
const int NUM_FIELDS = NUM_DESCRIPTORS * 3 + 2;
const int MIN_LENGTH_LINE = NUM_FIELDS;

Mat adapt_gauss_threshold(Mat image, int blur_size, int threshold_type, int block_size, 
    double c);
Mat adapt_mean_threshold(Mat image, int blur_size, int threshold_type, int block_size,
    double c);
Mat otsu_threshold(Mat image, int gauss_size);

//si no existe el fichero devuelve false
void read_data(const string& file_name, const string& obj_name, double* data, vector<string>& lines);

//filtra los contornos detectados que no se corresponden con el objeto
int filter_contours(const vector<vector<Point>> contours);

void calculate_descriptors(const int theBiggest, const vector<vector<Point>> contours,
  float& area, float& perim, double huMoments[]);

//Acumular numeradores de areas, perimetros y momentos
void train_model(double data[], const float area, const float perim, const double huMoments[]);

void write_data(const string& model_file_name, const string& obj_name,
  vector<string> lines, const double data[]);

//Formato del fichero de aprendizaje:
//<nombre_de_la_figura>;<n>;<media1>;<varianza1>;<varianza_estimada1>;<media2>;...<varianza_estimada5>
//<nombre_de_la_figura2>;...
//...
// Siendo 1) area 2) perimetro 3) momento1 4) momento2 5) momento3
int main(int argc, char** argv){
  if (argc != 3){ // Check parameters
    cout << "Wrong number of parameters, usage:\n";
    cout << "aprender <nomfich> <nomobj>\n";
    exit(1);
  }

  // Lectura de parametros
  string fich_name, obj_name;
  fich_name = PATH + argv[1];
  obj_name = argv[2];
  //fich_name = PATH + "rectangulo5.pgm";
  //obj_name = "rectangulo";

  // Lee el archivo a aprender
  Mat image = imread(samples::findFile(fich_name), IMREAD_GRAYSCALE);
  if(image.empty()){
    printf("Error opening image: %s\n", fich_name.c_str());
    return EXIT_FAILURE;
  }
  imshow("Image to learn", image);

  //Leer el fichero del modelo
  vector<string> lines;
  double data[NUM_FIELDS];
  string model_file_name = PATH + DATA_NAME;
  read_data(model_file_name, obj_name, data, lines);

  //Otsu threshold
  Mat otsu = otsu_threshold(image, 3);
  //Mat otsu =
  //  adapt_mean_threshold(image, 7, THRESH_BINARY_INV, 51, 5);
  imshow("Image otsurized", otsu);

  //Find contours
  //Mat canny;
  //Canny(otsu, canny, 100, 200);
  vector<vector<Point>> contours;
  findContours(otsu, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

  int theBiggest = filter_contours(contours);
  
  float area, perim;
  double huMoments[7];
  calculate_descriptors(theBiggest, contours, area, perim, huMoments);
  
  train_model(data, area, perim, huMoments);
  
  // Guarda la info en el fichero objetos
  write_data(model_file_name, obj_name, lines, data);

  waitKey(0);

  return EXIT_SUCCESS;
}

Mat adapt_gauss_threshold(Mat image, int blur_size, int threshold_type, int block_size, 
    double c){
  Mat result = image.clone();
  medianBlur(result, result, blur_size);
  adaptiveThreshold(result, result, 255, ADAPTIVE_THRESH_GAUSSIAN_C, threshold_type, 
      block_size, c);
  return result;
}

Mat adapt_mean_threshold(Mat image, int blur_size, int threshold_type, int block_size,
  double c){
  Mat result = image.clone();
  medianBlur(result, result, blur_size);
  adaptiveThreshold(result, result, 255, ADAPTIVE_THRESH_MEAN_C, threshold_type,
      block_size, c);
  return result;
}

Mat otsu_threshold(Mat image, int gauss_size){
  Mat result = image.clone();
  GaussianBlur(result, result, Size(gauss_size, gauss_size), 0, 0);
  threshold(result, result, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);
  return result;
}

void read_data(const string& file_name, const string& obj_name, double* data, vector<string>& lines){
  std::ifstream objects(file_name);

  // Lee datos
  char* buffer = new char[BUFF_LENGTH];
  string buffer_s;
  int readedCount;

  int pos;

  if (objects.is_open()) { // Existen datos anteriores
    do {
      objects.read(buffer, BUFF_LENGTH);
      readedCount = objects.gcount();
      buffer_s = buffer_s + string(buffer).substr(0, readedCount);
    } while (readedCount == BUFF_LENGTH);
    // Los divide en lineas
    pos = buffer_s.find("\n");
    while (pos != string::npos) {
      if (pos > MIN_LENGTH_LINE) {
        lines.push_back(buffer_s.substr(0, pos));
      }
      buffer_s.erase(0, pos + 1);
      pos = buffer_s.find("\n");
    }
    std::ofstream s(PATH + DATA_NAME, std::ofstream::trunc);
    s.close();
  }
  else { // Primer objeto aprendido
    cout << "objetos.txt not found, creating a new file\n";
    std::ofstream s(PATH + DATA_NAME);
    s.close();
    data[0] = 0;
  }
  objects.close();


  // Encuentra la clase afectada, o no!
  int required = -1;
  for (int i = 0; i < lines.size(); i++) {
    pos = lines[i].find(";");
    if (lines[i].substr(0, pos) == obj_name) { // Encontrado
      required = i;
      break;
    }
  }

  if (required == -1) { // No encontrado, crea nuevo
    required = lines.size();
    lines.push_back(obj_name + ";0;0;0;0;0;0;0;0;0;0;0");
  }

  // Lee datos de la clase afectada
  string required_Class = lines[required];
  lines.erase(lines.begin() + required);
  pos = required_Class.find(";");
  required_Class.erase(0, pos + 1);
  for (int i = 0; i < NUM_FIELDS; i++) {
    pos = required_Class.find(";");
    data[i] = stod(required_Class.substr(0, pos));
    required_Class.erase(0, pos + 1);
  }
}

int filter_contours(const vector<vector<Point>> contours) {
  float maxArea = 0;
  float aux;
  int theBiggest = 0;
  for (int i = 0; i < contours.size(); i++) {
    aux = contourArea(contours[i]);
    if (aux > maxArea) {
      theBiggest = i;
      maxArea = aux;
    }
  }
  return theBiggest;
}

void calculate_descriptors(const int theBiggest, const vector<vector<Point>> contours,
  float& area, float& perim, double huMoments[]) {
  area = contourArea(contours[theBiggest]);
  perim = arcLength(contours[theBiggest], true);
  Moments mu = moments(contours[theBiggest]); // Los momentos
  HuMoments(mu, huMoments);
}

void train_model(double data[], const float area, const float perim, const double huMoments[]) {
  // El primer campo es el numero de elementos
  if (data[0] == 0) {
    data[0] = 1; // Numero de muestras
    data[1] = area; // Media del area
    data[2] = 0; // Varianza del area
    data[3] = 0; // Varianza estimada del area
    data[4] = perim; // Media del perimetro
    data[5] = 0; // Varianza del perimetro
    data[6] = 0; // Varianza estimada del perimetro
    data[7] = huMoments[0]; // Media del primer momento Hu
    data[8] = 0; // Varianza del primer momento Hu
    data[9] = 0; // Varianza estimada del primer momento Hu
    data[10] = huMoments[1]; // Media del segundo momento Hu
    data[11] = 0; // Varianza del segundo momento Hu
    data[12] = 0; // Varianza estimada del segundo momento Hu
    data[13] = huMoments[2]; // Media del tercer momento Hu
    data[14] = 0; // Varianza del tercer momento Hu
    data[15] = 0; // Varianza estimada del tercer momento Hu
  }
  else {
    // Numero de muestras
    data[0]++;
    // Numerador de medias del area
    data[1] = data[1] + area;
    // Numerador de varianzas del area
    data[2] = data[2] + pow((area - data[1] / data[0]), 2);
    // Numerador de varianzas estimadas del area
    data[3] = pow((data[1] / data[0] * VAR_PERCENTAGE), 2) + (data[0] - 1) * data[2];
    // Numerador de medias del perimetro
    data[4] = data[4] + perim;
    // Numerador de varianzas del perimetro
    data[5] = data[5] + pow((perim - data[4] / data[0]), 2);
    // Numerador de varianzas estimadas del perimetro
    data[6] = pow((data[4] / data[0] * VAR_PERCENTAGE), 2) + (data[0] - 1) * data[5];
    // Numerador de medias del primer momento Hu
    data[7] = data[7] + huMoments[0];
    // Numerador de varianzas del primer momento Hu
    data[8] = data[8] + pow((huMoments[0] - data[7] / data[0]), 2);
    // Numerador de varianzas estimadas del primer momento Hu
    data[9] = pow((data[7] / data[0] * VAR_PERCENTAGE), 2) + (data[0] - 1) * data[8];
    // Numerador de medias del segundo momento Hu
    data[10] = data[10] + huMoments[1];
    // Numerador de varianzas del segundo momento Hu
    data[11] = data[11] + pow((huMoments[1] - data[10] / data[0]), 2);
    // Numerador de varianzas estimadas del segundo momento Hu
    data[12] = pow((data[10] / data[0] * VAR_PERCENTAGE), 2) + (data[0] - 1) * data[11];
    // Numerador de medias del tercer momento Hu
    data[13] = data[13] + huMoments[2];
    // Numerador de varianzas del tercer momento Hu
    data[14] = data[14] + pow((huMoments[2] - data[13] / data[0]), 2);
    // Numerador de varianzas estimadas del tercer momento Hu
    data[15] = pow((data[13] / data[0] * VAR_PERCENTAGE), 2) + (data[0] - 1) * data[14];
  }
}
void write_data(const string& model_file_name, const string& obj_name, 
  vector<string> lines, const double data[]) {
  
  std::ofstream newObjects(PATH + DATA_NAME);
  for (int i = 0; i < lines.size(); i++) {
    newObjects << lines[i] << endl;
  }
  newObjects << obj_name;
  for (int i = 1; i < NUM_FIELDS; i++) {
    newObjects << ";" << data[i - 1];
  }
  newObjects << endl;
  newObjects.close();
}
