#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;

using std::size;
using std::string;
using std::vector;
using std::to_string;
using xfeatures2d::SURF;

const string CURVY_WALL[5] = {"00.jpeg", "01.jpeg", "02.jpeg", "03.jpeg", "04.jpeg"};
const string STRAIGHT_WALL[5] = {"10.jpeg", "11.jpeg", "12.jpeg", "13.jpeg", "14.jpeg"};
const string MURAL[2] = {"40.jpeg", "41.jpeg"}; //, "42.jpeg"};
const string HORIZONTAL_FROG[2] = { "50.jpeg", "51.jpeg" };
const string VERTICAL_TOWER[2] = { "20.jpeg", "21.jpeg" };
const string HORIZONTAL_CITY[2] = { "60.jpeg", "61.jpeg" };
const string HORIZONTAL_BUILDING[5] = { "70.JPG", "71.JPG", "72.JPG", "73.JPG", "74.JPG" };

// Lo cambiable:
const string IMAGE_SET[5] = { HORIZONTAL_BUILDING[0], HORIZONTAL_BUILDING[1], 
    HORIZONTAL_BUILDING[2], HORIZONTAL_BUILDING[3], HORIZONTAL_BUILDING[4] };
const string IMAGE_SET_3D[2] = { HORIZONTAL_CITY[0], HORIZONTAL_CITY[1] };
const string PATH = "C:\\Users\\pica\\Documents\\GitHub\\super-duper-system\\P4\\images\\";
const int WINDOWS_X[6] = {0, 600, 1200, 0, 600, 1200};
const int WINDOWS_Y[6] = {0, 0, 0, 600, 600, 600};

void orbHarris(Mat image1, Mat image2, float reject_ratio, float scale_factor);
void orbFAST(Mat image1, Mat image2, float reject_ratio, float scale_factor);
void sift(Mat image1, Mat image2, float reject_ratio);
void surf(Mat image1, Mat image2, float reject_ratio);
void akaze(Mat image1, Mat image2, float reject_ratio);

int main(){
  // Load and show images
  string window_name;
  string fich_name = PATH + IMAGE_SET[0];
  Mat image1, image2, image3d1, image3d2;
  image1 = imread(samples::findFile(fich_name), IMREAD_COLOR);
  if (image1.empty()){
    printf("Error opening image: %s\n", fich_name.c_str());
    return EXIT_FAILURE;
  }
  resize(image1, image1, Size(512, 384)); // Originalmente son 2048x1536

  fich_name = PATH + IMAGE_SET[1];
  image2 = imread(samples::findFile(fich_name), IMREAD_COLOR);
  if(image2.empty()){
    printf("Error opening image: %s\n", fich_name.c_str());
    return EXIT_FAILURE;
  }
  resize(image2, image2, Size(512, 384)); // Originalmente son 2048x1536

  //waitKey(0);

  float reject_ratio = 0.8;
  Mat Panorama = addPanorama(image1, image2, reject_ratio);
  imshow("asd", Panorama);
  waitKey(0);

  return EXIT_SUCCESS;
}

Mat addPanorama(Mat orig, Mat added, float reject_ratio){
  Mat clone1 = orig.clone();
  Mat clone2 = added.clone();
  Mat gray1, gray2, surfResult, result;
  Ptr<SURF> surf = SURF::create();
  vector<KeyPoint> key_points1, key_points2;
  Mat descriptors1, descriptors2;
  vector<vector<DMatch>> matches;
  vector<DMatch> filtered_matches;

  string window_name = "Surf con ratio " + to_string(reject_ratio);

  cvtColor(clone1, gray1, COLOR_BGR2GRAY);
  cvtColor(clone2, gray2, COLOR_BGR2GRAY);

  surf->detectAndCompute(gray1, surfResult, key_points1, descriptors1);
  surf->detectAndCompute(gray2, surfResult, key_points2, descriptors2);

  Ptr<BFMatcher> bf = BFMatcher::create(NORM_L2, false);
  bf->knnMatch(descriptors1, descriptors2, matches, 2);
  for(int i = 0; i < matches.size(); i++){
    if(matches[i][0].distance < matches[i][1].distance * reject_ratio){
      filtered_matches.push_back(matches[i][0]);
    }
  }

  drawMatches(clone1, key_points1, clone2, key_points2, filtered_matches, result);
  resize(result, result, Size(1024, 384));
  imshow(window_name, result);
  moveWindow(window_name, WINDOWS_X[3], WINDOWS_Y[3]);

  vector<Point2f> matched_key_points1;
  vector<Point2f> matched_key_points2;
  for(size_t i = 0; i < filtered_matches.size(); i++){
    matched_key_points1.push_back(key_points1[filtered_matches[i].queryIdx].pt);
    matched_key_points2.push_back(key_points2[filtered_matches[i].trainIdx].pt);
  }
  Mat H = findHomography(matched_key_points1, matched_key_points2, RANSAC);

  std::vector<Point2f> cuatroPuntos;
  cuatroPuntos.push_back(Point2f(0, 0));
  cuatroPuntos.push_back(Point2f(clone1.size().width, 0));
  cuatroPuntos.push_back(Point2f(0, clone1.size().height));
  cuatroPuntos.push_back(Point2f(clone1.size().width, clone1.size().height));


  Mat MDestino;
  perspectiveTransform(Mat(cuatroPuntos), MDestino, H);

  // Calcular esquinas de imagen 2
  double min_x, min_y, tam_x, tam_y;
  float min_x1, min_x2, min_y1, min_y2, max_x1, max_x2, max_y1, max_y2;
  min_x1 = min(MDestino.at<Point2f>(0).x, MDestino.at<Point2f>(1).x);
  min_x2 = min(MDestino.at<Point2f>(2).x, MDestino.at<Point2f>(3).x);
  min_y1 = min(MDestino.at<Point2f>(0).y, MDestino.at<Point2f>(1).y);
  min_y2 = min(MDestino.at<Point2f>(2).y, MDestino.at<Point2f>(3).y);
  max_x1 = max(MDestino.at<Point2f>(0).x, MDestino.at<Point2f>(1).x);
  max_x2 = max(MDestino.at<Point2f>(2).x, MDestino.at<Point2f>(3).x);
  max_y1 = max(MDestino.at<Point2f>(0).y, MDestino.at<Point2f>(1).y);
  max_y2 = max(MDestino.at<Point2f>(2).y, MDestino.at<Point2f>(3).y);
  min_x = min(min_x1, min_x2);
  min_y = min(min_y1, min_y2);
  tam_x = max(max_x1, max_x2);
  tam_y = max(max_y1, max_y2);

  // Matriz de transformación
  Mat Htr = Mat::eye(3, 3, CV_64F);
  if(min_x < 0){
    tam_x = clone2.size().width - min_x;
    Htr.at<double>(0, 2) = -min_x;
  }
  if(min_y < 0){
    tam_y = clone2.size().height - min_y;
    Htr.at<double>(1, 2) = -min_y;
  }

  // Construir panorama
  Mat Panorama;
  Panorama = Mat(Size(tam_x, tam_y), CV_32F);
  warpPerspective(clone2, Panorama, Htr, Panorama.size(), INTER_LINEAR, BORDER_CONSTANT, 0);
  warpPerspective(clone1, Panorama, (Htr * H), Panorama.size(), INTER_LINEAR, BORDER_TRANSPARENT, 0);
}

void orbHarris(Mat image1, Mat image2, float reject_ratio, float scale_factor){
  Mat clone1 = image1.clone();
  Mat clone2 = image2.clone();
  Mat gray1, gray2, orbResult, result;
  Ptr<ORB> orbHarris = ORB::create(500, scale_factor, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
  vector<KeyPoint> key_points1, key_points2;
  Mat descriptors1, descriptors2;
  string window_name = "Orb Harris con ratio " + to_string(reject_ratio);
  vector<vector<DMatch>> matches;
  vector<DMatch> filtered_matches;

  cvtColor(clone1, gray1, COLOR_BGR2GRAY);
  cvtColor(clone2, gray2, COLOR_BGR2GRAY);

  orbHarris->detectAndCompute(gray1, orbResult, key_points1, descriptors1);
  orbHarris->detectAndCompute(gray2, orbResult, key_points2, descriptors2);

  Ptr<BFMatcher> bf = BFMatcher::create(NORM_L2, false);
  bf->knnMatch(descriptors1, descriptors2, matches, 2);
  for (int i = 0; i < matches.size(); i++){
    if (matches[i][0].distance < matches[i][1].distance * reject_ratio){
      filtered_matches.push_back(matches[i][0]);
    }
  }

  drawMatches(image1, key_points1, image2, key_points2, filtered_matches, result);
  resize(result, result, Size(1024, 384));
  imshow(window_name, result);
  moveWindow(window_name, WINDOWS_X[3], WINDOWS_Y[3]);

}

void orbFAST(Mat image1, Mat image2, float reject_ratio, float scale_factor){
  Mat clone1 = image1.clone();
  Mat clone2 = image2.clone();
  Mat gray1, gray2, orbResult, result;
  Ptr<ORB> orbFAST = ORB::create(500, scale_factor, 8, 31, 0, 2, ORB::FAST_SCORE, 31, 20);
  vector<KeyPoint> key_points1, key_points2;
  Mat descriptors1, descriptors2;
  string window_name = "Orb FAST con ratio " + to_string(reject_ratio);
  vector<vector<DMatch>> matches;
  vector<DMatch> filtered_matches;

  cvtColor(clone1, gray1, COLOR_BGR2GRAY);
  cvtColor(clone2, gray2, COLOR_BGR2GRAY);

  orbFAST->detectAndCompute(gray1, orbResult, key_points1, descriptors1);
  orbFAST->detectAndCompute(gray2, orbResult, key_points2, descriptors2);

  Ptr<BFMatcher> bf = BFMatcher::create(NORM_L2, false);
  bf->knnMatch(descriptors1, descriptors2, matches, 2);
  for(int i = 0; i < matches.size(); i++){
    if(matches[i][0].distance < matches[i][1].distance * reject_ratio){
      filtered_matches.push_back(matches[i][0]);
    }
  }

  drawMatches(image1, key_points1, image2, key_points2, filtered_matches, result);
  resize(result, result, Size(1024, 384));
  imshow(window_name, result);
  moveWindow(window_name, WINDOWS_X[3], WINDOWS_Y[3]);
}

void sift(Mat image1, Mat image2, float reject_ratio){
  Mat clone1 = image1.clone();
  Mat clone2 = image2.clone();
  Mat gray1, gray2, siftResult, result;
  Ptr<SIFT> sift = SIFT::create();
  vector<KeyPoint> key_points1, key_points2;
  Mat descriptors1, descriptors2;
  string window_name = "Sift con ratio " + to_string(reject_ratio);
  vector<vector<DMatch>> matches;
  vector<DMatch> filtered_matches;

  cvtColor(clone1, gray1, COLOR_BGR2GRAY);
  cvtColor(clone2, gray2, COLOR_BGR2GRAY);
  sift->detectAndCompute(gray1, siftResult, key_points1, descriptors1);
  sift->detectAndCompute(gray2, siftResult, key_points2, descriptors2);

  Ptr<BFMatcher> bf = BFMatcher::create(NORM_L2, false);
  bf->knnMatch(descriptors1, descriptors2, matches, 2);
  for(int i = 0; i < matches.size(); i++){
    if(matches[i][0].distance < matches[i][1].distance * reject_ratio){
      filtered_matches.push_back(matches[i][0]);
    }
  }

  drawMatches(image1, key_points1, image2, key_points2, filtered_matches, result);
  resize(result, result, Size(1024, 384));
  imshow(window_name, result);
  moveWindow(window_name, WINDOWS_X[3], WINDOWS_Y[3]);
}

void surf(Mat image1, Mat image2, float reject_ratio){
  Mat clone1 = image1.clone();
  Mat clone2 = image2.clone();
  Mat gray1, gray2, surfResult, result;
  Ptr<SURF> surf = SURF::create();
  vector<KeyPoint> key_points1, key_points2;
  Mat descriptors1, descriptors2;
  string window_name = "Surf con ratio " + to_string(reject_ratio);
  vector<vector<DMatch>> matches;
  vector<DMatch> filtered_matches;

  cvtColor(clone1, gray1, COLOR_BGR2GRAY);
  cvtColor(clone2, gray2, COLOR_BGR2GRAY);

  surf->detectAndCompute(gray1, surfResult, key_points1, descriptors1);
  surf->detectAndCompute(gray2, surfResult, key_points2, descriptors2);

  Ptr<BFMatcher> bf = BFMatcher::create(NORM_L2, false);
  bf->knnMatch(descriptors1, descriptors2, matches, 2);
  for(int i = 0; i < matches.size(); i++){
    if(matches[i][0].distance < matches[i][1].distance * reject_ratio){
      filtered_matches.push_back(matches[i][0]);
    }
  }

  drawMatches(image1, key_points1, image2, key_points2, filtered_matches, result);
  resize(result, result, Size(1024, 384));
  imshow(window_name, result);
  moveWindow(window_name, WINDOWS_X[3], WINDOWS_Y[3]);
}

void akaze(Mat image1, Mat image2, float reject_ratio){
  Mat clone1 = image1.clone();
  Mat clone2 = image2.clone();
  Mat gray1, gray2, akazeResult, result;
  Ptr<AKAZE> akaze = AKAZE::create();
  vector<KeyPoint> key_points1, key_points2;
  Mat descriptors1, descriptors2;
  string window_name = "Akaze con ratio " + to_string(reject_ratio);
  vector<vector<DMatch>> matches;
  vector<DMatch> filtered_matches;

  cvtColor(clone1, gray1, COLOR_BGR2GRAY);
  cvtColor(clone2, gray2, COLOR_BGR2GRAY);

  akaze->detectAndCompute(gray1, akazeResult, key_points1, descriptors1);
  akaze->detectAndCompute(gray2, akazeResult, key_points2, descriptors2);

  Ptr<BFMatcher> bf = BFMatcher::create(NORM_L2, false);
  bf->knnMatch(descriptors1, descriptors2, matches, 2);
  for(int i = 0; i < matches.size(); i++){
    if(matches[i][0].distance < matches[i][1].distance * reject_ratio){
      filtered_matches.push_back(matches[i][0]);
    }
  }

  drawMatches(image1, key_points1, image2, key_points2, filtered_matches, result);
  resize(result, result, Size(1024, 384));
  imshow(window_name, result);
  moveWindow(window_name, WINDOWS_X[3], WINDOWS_Y[3]);
}