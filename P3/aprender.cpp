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
  if (argc != 3){ // Check parameters
    cout << "Wrong number of parameters, usage:\n";
    cout << "aprender <nomfich> <nomobj>\n";
    exit(1);
  }

  // Parameters reading
  string fich_name = argv[1];
  string obj_name = argv[2];
  
  fich_name = PATH + "circulo1.pgm";

  // File reading
  Mat image = imread(samples::findFile(fich_name), IMREAD_COLOR);
  if (image.empty()){
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

  // Descarta basura
  float maxArea = 0;
  float aux;
  int theBiggest = 0;
  for (int i = 0; i < contours.size(); i++){
    aux = contourArea(contours[i]);
    if (aux > maxArea){
      theBiggest = i;
      maxArea = aux;
    }
  }

  Moments mu = moments(contours[theBiggest]); // Los momentos
  float area = maxArea;
  float perim = arcLength(contours[theBiggest], true);

  std::ifstream objects(PATH + DATA_NAME);

  // Lee datos
  char* buffer = new char[BUFF_LENGTH];
  string buffer_s;
  int readedCount;
  if (objects.is_open()){ // Existen datos anteriores
    do{
      objects.read(buffer, BUFF_LENGTH);
      readedCount = objects.gcount();
      buffer_s = buffer_s + string(buffer).substr(0, readedCount);
    }while(readedCount == BUFF_LENGTH);
  }else{ // Primer objeto aprendido
    cout << "No existe objetos.txt, creando uno nuevo\n";
    std::ofstream s(PATH + DATA_NAME);
    s.close();
  }

  // Los divide en lineas
  int pos = buffer_s.find("\n");
  vector<string> lines;
  while (pos != string::npos){
    if (pos > MIN_LENGTH_LINE){
      lines.push_back(buffer_s.substr(0, pos));
    }
    buffer_s.erase(0, pos + 1);
    pos = buffer_s.find("\n");
  }
  //cout << lineas.size() << endl;
  //for (int i = 0; i < lineas.size(); i++){
  //  cout << lineas[i] << endl;
  //}

  // Encuentra la clase afectada, o no!
  int required = -1;
  for (int i = 0; i < lines.size(); i++){
    pos = lines[i].find(";");
    if (lines[i].substr(0, pos) == obj_name){ // Encontrado
      required = i;
      break;
    }
  }

  if (required != -1){ // No encontrado, creamos nuevo
    required = lines.size();
    lines.push_back(obj_name + ";0;0;0;0;0;0;0;0;0;0;0");
  }

  // Lee datos de la clase afectada
  string required_Class = lines[required];
  lines.erase(lines.begin() + required);
  pos = required_Class.find(";");
  required_Class.erase(0, pos + 1);
  double data[11];
  //for (int i = 0; i < NUM_FIELDS; i++){
  //  data[i] = 
  //}
  
  // SEGUIR DESDE AQUI
  //float media = muestras / nMuestras;
  //float varianza = sum(pow((muestra - media), 2)) / (nMuestras);

  waitKey(0);

  return EXIT_SUCCESS;
}
  /*
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
  int nLabels = connectedComponents(otsu, labelImage, 4);
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

  // Descartamos basurilla
  vector<vector<Point>> nice_contours;
  for(size_t i = 0; i < contours.size(); i++){
    if ((contourArea(contours[i]) > 100) && (arcLength(contours[i], true) > 100)){
      nice_contours.push_back(contours[i]);
    }
  }

  // live the past, forget the moment, o algo así
  vector<Moments> mu(nice_contours.size());
  for(size_t i = 0; i < nice_contours.size(); i++){
    mu[i] = moments(nice_contours[i]);
  }

  // Me ha costado pero esto es los moment center
  vector<Point2f> mc(nice_contours.size());
  for(size_t i = 0; i < nice_contours.size(); i++){
    //add 1e-5 to avoid division by zero
    mc[i] = Point2f(static_cast<float>(mu[i].m10 / (mu[i].m00 + 1e-5)),
        static_cast<float>(mu[i].m01 / (mu[i].m00 + 1e-5)));
    cout << "mc[" << i << "]=" << mc[i] << endl;
  }
  
  // las dibujaciones
  Mat drawing = Mat::zeros(canny.size(), CV_8UC3);
  for(size_t i = 0; i < nice_contours.size(); i++){
    Scalar color = Scalar(rand() & 255, rand() & 255, rand() & 255);
    drawContours(drawing, nice_contours, (int)i, color, 2);
    circle(drawing, mc[i], 4, color, -1);
  }

  // las imprimiciones
  imshow("Contours", drawing);
  cout << "\t Info: Area and Contour Length \n";
  for(size_t i = 0; i < nice_contours.size(); i++){
    cout << " * Contour[" << i << "] - Area (M_00) = " << std::fixed << std::setprecision(2) << mu[i].m00
        << " - Area OpenCV: " << contourArea(nice_contours[i]) << " - Length: " << arcLength(nice_contours[i], true) << endl;
  }

  waitKey(0);

  return EXIT_SUCCESS;
}
*/
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
