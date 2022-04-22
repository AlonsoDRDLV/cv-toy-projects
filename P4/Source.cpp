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

// Lo cambiable:
const string PATH = "C:\\Users\\pica\\Documents\\GitHub\\super-duper-system\\P4\\images\\";
const string* IMAGE_SET = HORIZONTAL_FROG;
const int IMAGE_SET_LENGTH = 2;
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
  Mat image1, image2, result;
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

  window_name = "Image 1";
  imshow(window_name, image1);
  moveWindow(window_name, WINDOWS_X[0], WINDOWS_Y[0]);

  window_name = "Image 2";
  imshow(window_name, image2);
  moveWindow(window_name, WINDOWS_X[1], WINDOWS_Y[1]);

  waitKey(0);

  // ORB HARRIS
  orbHarris(image1, image2, 0.8, 1.2);
  waitKey(0);

  // ORB FAST
  orbFAST(image1, image2, 0.7, 1.1);
  waitKey(0);

  // SIFT
  sift(image1, image2, 0.5);
  waitKey(0);

  // SURF
  surf(image1, image2, 0.6);
  waitKey(0);

  // AKAZE
  akaze(image1, image2, 0.7);
  waitKey(0);

  return EXIT_SUCCESS;
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
  imshow(window_name, result);
  moveWindow(window_name, WINDOWS_X[3], WINDOWS_Y[3]);
}