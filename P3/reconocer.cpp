#include "opencv2/imgcodecs.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include <vector>

using std::to_string;
using std::string;
using std::vector;
using std::cin;
using std::cout;
using std::endl;
using namespace cv;

const string PATH = "C:\\Users\\pica\\Documents\\GitHub\\super-duper-system\\P3\\images\\";
const string DATA_NAME = "objetos.txt";
const int MIN_AREA = 100;
const int MIN_PERIM = 100;
const int HU_MOMENTS = 3;
const int NUM_DESCRIPTORS = 5;
const int NUM_FIELDS = NUM_DESCRIPTORS * 3 + 2;
const int MIN_LENGTH_LINE = NUM_FIELDS;
const double chi_square_value = 11.07; //para grado de libertad = NUM_DESCRIPTORS y alfa = 0.05

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
  if (argc != 2){ // Check parameters
    cout << "Wrong number of parameters, usage:\n";
    cout << "reconocer <nomfich>\n";
    exit(1);
  }

  // Lectura de parametros
  string fich_name = PATH + argv[1];
  string objects_name = PATH + DATA_NAME;

  // Lee datos
  string buffer_s;
  int readedCount, pos;
  vector<string> lines;
  vector<string> classes;
  vector<int> n; //numero muestras con las que se ha entrenado al modelo para una clase concreta
  vector<vector<double>> means; //medias de los descriptores de cada clase
  vector<vector<double>> variances; //varianzas de los descriptores de cada clase
  vector<vector<double>> estimated_variances; //varianzas estimadas de los descriptores de cada clase
  
  std::ifstream objects(objects_name);

  if (objects.is_open()){ // Se encuentran datos anteriores
    for (string line; std::getline(objects, line); ){
      std::stringstream line_stream(line);
      vector<double> means_aux;
      vector<double> variances_aux;
      vector<double> estimated_variances_aux;
      int i = 0;
      for (string token; std::getline(line_stream, token, ';'); i++){
        if(i == 0){
          classes.push_back(token);
        }else if(i == 1){
          n.push_back(stoi(token));
        }else if((i % 3) == 2){
          //soy la buena, descomentame cuando compruebes que lee bien
          means_aux.push_back(stod(token)/n[n.size() - 1]);
          //means_aux.push_back(stod(token)); 
        }else if((i % 3) == 0) {
          variances_aux.push_back(stod(token)/n[n.size() - 1]);
          //variances_aux.push_back(stod(token));
        }else if ((i % 3) == 1) {
          estimated_variances_aux.push_back(stod(token)/n[n.size() - 1]);
          //estimated_variances_aux.push_back(stod(token));
        }
      }
      //borrame despues de comprobar que va
      cout << line << endl;
      for (double a : means_aux) cout << a << " "; cout << endl;
      for (double a : variances_aux) cout << a << " "; cout << endl;
      for (double a : estimated_variances_aux) cout << a << " "; cout << endl;
      
      means.push_back(means_aux);
      variances.push_back(variances_aux);
      estimated_variances.push_back(estimated_variances_aux);
    }
    objects.close();
  }else{ // No se encuentra el fichero con los datos
    cout << "objetos.txt not found, exiting\n";
    exit(1);
  }

  // Lee la imagen a clasificar
  Mat image = imread(samples::findFile(fich_name), IMREAD_GRAYSCALE);
  if(image.empty()){
    printf("Error opening image: %s\n", fich_name.c_str());
    return EXIT_FAILURE;
  }
  imshow("Image to learn", image);

  // Aplica thresholding de Otsu para pasarla a monocromo: fondo-negro, resto-blanco
  Mat otsu = otsu_threshold(image, 3);
  imshow("Image otsurized", otsu);

  Mat canny; // Ni siquiera se usaba esto en ningun momento
  Canny(otsu, canny, 100, 200);
  // Pero a mi compa le gusta pensar que si asi que lo dejo

  // ANTES:
  //findContours(canny, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
  // DESPUÉS:
  vector<vector<Point>> contours;
  vector<vector<Point>> contour;
  // Saca la línea azul esa fea a la izquierda, tengo que ver cómo retirarla
  Mat labelImage(otsu.size(), CV_32S);
  int nLabels = connectedComponents(otsu, labelImage, 8);
  Mat* comp = new Mat[nLabels];
  for(int i = 1; i < nLabels; i++){
    comp[i - 1] = Mat(otsu.size(), CV_8UC1, Scalar(0));
    for(int r = 0; r < otsu.rows; ++r){
      for(int c = 0; c < otsu.cols; ++c){
        int pixel = labelImage.at<int>(r, c);
        if(pixel == i){
          comp[i - 1].at<uchar>(r, c) = 255;
        }else{
          comp[i - 1].at<uchar>(r, c) = 0;
        }
      }
    }
    imshow("Connected Components " + to_string(i - 1), comp[i - 1]);
    findContours(comp[i - 1], contour, RETR_TREE, CHAIN_APPROX_SIMPLE);
    if ((contourArea(contour[0]) > MIN_AREA) && (arcLength(contour[0], true) > MIN_PERIM)){
      //cout << contourArea(contour[0]) << " " << arcLength(contour[0], true) << endl;
      contours.push_back(contour[0]);
    }
  }

  // Saca los valores de los descriptores de todos los objetos detectados
  vector<vector<double>> detected_obj_descriptors; //area, perimetro, momentos 
  vector<double> v_aux;
  for (int i = 0; i < contours.size(); i++){
    v_aux.push_back(contourArea(contours[i]));
    v_aux.push_back(arcLength(contours[i], true));
    Moments mu = moments(contours[i]);  
    double huMoments[7];
    HuMoments(mu, huMoments);
    for (int moment = 0; moment < HU_MOMENTS; moment++){
      //huMoments[moment] = -1 * copysign(1.0, huMoments[moment]) 
      //    * log10(abs(huMoments[moment]));
      v_aux.push_back(huMoments[moment]);
    }
    detected_obj_descriptors.push_back(v_aux);
    v_aux.clear();
  }

  // Calcula las distancias entre todas las clases para todos los contornos
  // Cada componente: {Distancias de un objeto a todas las clases}
  vector<vector<double>> mahalanobis_distances;
  for (int i_obj = 0; i_obj < detected_obj_descriptors.size(); i_obj++){
    vector<double> distances_to_classes;
    for (int i_class = 0; i_class < means.size(); i_class++){ //means.size() y variance.size() == numero de clases
      double distance = 0.0;
      for (int i_desc = 0; i_desc < means[0].size(); i_desc++){ //los elementos de means y variances tienen size() == num de descriptores
        double mean, variance, descriptor;
        descriptor = detected_obj_descriptors[i_obj][i_desc];
        mean = means[i_class][i_desc];
        variance = estimated_variances[i_class][i_desc];
        cout << mean << " " << variance << " i class " << i_class << " i_desc " << i_desc << "     ";
        //cout << i_class << "  descriptor: " << descriptor << "  media: " << mean << "  varianza: " << variance << endl;
        distance += pow((descriptor - mean), 2) / variance;
      }
      distances_to_classes.push_back(distance);
      cout << endl;
    }
    mahalanobis_distances.push_back(distances_to_classes);
  }

  // Checkear distancias
  for (vector<double> a : mahalanobis_distances){
    int i = 0;
    for (double b : a){
      cout << "distance: " << b << " - ";
      if (b < chi_square_value){
        cout << classes[i]; 
      }
      i++;
    }
    cout << endl;
  }

  /*
  // Por ultimo se comprueba si para cada contorno se le asocia una clase
  vector<vector<bool>> chisquare_hypothesis_test;
  for (vector<double> distances_to_classes : mahalanobis_distances) {
    vector<bool> obj_classification; //almacena si pertenece o no el objeto a las clases
    for (double distance : distances_to_classes) {
      obj_classification.push_back(distance <= chi_square_value);
    }
    chisquare_hypothesis_test.push_back(obj_classification);
  }
  
  //se imprimen los resultados:
  vector<Point2f> mc(contours.size());
  vector<Moments> mu(contours.size());
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
*/
  waitKey(0);
  
  return EXIT_SUCCESS;

}

Mat adapt_gauss_threshold(Mat image, int blur_size, int threshold_type, int block_size, 
    double c){
  Mat result = image.clone();
  medianBlur(result, result, blur_size);
  adaptiveThreshold(result, result, 255, ADAPTIVE_THRESH_GAUSSIAN_C, threshold_type, 
      block_size, c);
  result = Mat(result.size(), result.type(), 255) - result;
  return result;
}

Mat adapt_mean_threshold(Mat image, int blur_size, int threshold_type, int block_size,
  double c){
  Mat result = image.clone();
  medianBlur(result, result, blur_size);
  adaptiveThreshold(result, result, 255, ADAPTIVE_THRESH_MEAN_C, threshold_type,
      block_size, c);
  result = Mat(result.size(), result.type(), 255) - result;
  return result;
}

Mat otsu_threshold(Mat image, int gauss_size){
  Mat result = image.clone();
  GaussianBlur(result, result, Size(gauss_size, gauss_size), 0, 0);
  threshold(result, result, 0, 255, THRESH_BINARY + THRESH_OTSU);
  result = Mat(result.size(), result.type(), 255) - result;
  return result;
}
