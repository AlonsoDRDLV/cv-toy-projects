#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using std::size;
using std::string;
using std::vector;
using std::to_string;
using namespace cv;

const string PATH = "C:\\Users\\pica\\Documents\\GitHub\\super-duper-system\\P4\\images\\";
const string CURVY_WALL[5] = {"00.jpeg", "01.jpeg", "02.jpeg", "03.jpeg", "04.jpeg"};
const string STRAIGHT_WALL[5] = {"10.jpeg", "11.jpeg", "12.jpeg", "13.jpeg", "14.jpeg"};
const string MURAL[2] = {"40.jpeg", "41.jpeg"}; //, "42.jpeg"};
const string HORIZONTAL_FROG[2] = { "50.jpeg", "51.jpeg" };

// Lo cambiable:
const string* IMAGE_SET = HORIZONTAL_FROG;
const int IMAGE_SET_LENGTH = 2;

const int WINDOWS_X[6] = {0, 600, 1200, 0, 600, 1200};
const int WINDOWS_Y[6] = {0, 0, 0, 600, 600, 600};

Mat harris(Mat image, int threshold);
Mat orbHarris(Mat image);
Mat orbFAST(Mat image);
Mat sift(Mat image);
Mat surf(Mat image);
Mat akaze(Mat image);


int main(){
  string window_name;

  // Load images
  string fich_name = PATH + IMAGE_SET[0];
  Mat image1, image2, result;
  image1 = imread(samples::findFile(fich_name), IMREAD_COLOR);
  if (image1.empty()){
    printf("Error opening image: %s\n", fich_name.c_str());
    return EXIT_FAILURE;
  }
  resize(image1, image1, Size(512, 384)); // Originalmente son 2048x1536

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
  result = orbHarris(image1);
  window_name = "Orb Harris 1";
  imshow(window_name, result);
  moveWindow(window_name, WINDOWS_X[3], WINDOWS_Y[3]);

  result = orbHarris(image2);
  window_name = "Orb Harris 2";
  imshow(window_name, result);
  moveWindow(window_name, WINDOWS_X[4], WINDOWS_Y[4]);

  waitKey(0);

  // ORB FAST
  result = orbFAST(image1);
  window_name = "Orb FAST 1";
  imshow(window_name, result);
  moveWindow(window_name, WINDOWS_X[3], WINDOWS_Y[3]);

  result = orbFAST(image2);
  window_name = "Orb FAST 2";
  imshow(window_name, result);
  moveWindow(window_name, WINDOWS_X[4], WINDOWS_Y[4]);

  waitKey(0);

  // SIFT
  result = sift(image1);
  window_name = "Sift 1";
  imshow(window_name, result);
  moveWindow(window_name, WINDOWS_X[3], WINDOWS_Y[3]);

  result = sift(image2);
  window_name = "Sift 2";
  imshow(window_name, result);
  moveWindow(window_name, WINDOWS_X[4], WINDOWS_Y[4]);

  waitKey(0);

  // SURF
  result = surf(image1);
  window_name = "Surf 1";
  imshow(window_name, result);
  moveWindow(window_name, WINDOWS_X[3], WINDOWS_Y[3]);

  result = surf(image2);
  window_name = "Surf 2";
  imshow(window_name, result);
  moveWindow(window_name, WINDOWS_X[4], WINDOWS_Y[4]);

  waitKey(0);

  // AKAZE
  result = akaze(image1);
  window_name = "Akaze 1";
  imshow(window_name, result);
  moveWindow(window_name, WINDOWS_X[3], WINDOWS_Y[3]);
 
  result = akaze(image2);
  window_name = "Akaze 2";
  imshow(window_name, result);
  moveWindow(window_name, WINDOWS_X[4], WINDOWS_Y[4]);

  waitKey(0);

  return EXIT_SUCCESS;
}

Mat harris(Mat image, int threshold){
  Mat result = image.clone();
  int block_size = 2;
  int aperture_size = 5;
  double k = 0.04;
  Mat gray, harris, norm, scaled;
  cvtColor(result, gray, COLOR_BGR2GRAY);

  cornerHarris(gray, harris, block_size, aperture_size, k);

  normalize(harris, norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

  convertScaleAbs(norm, scaled);

  vector<KeyPoint> key_points;

  for(int i = 0; i < result.rows; i++){
    for(int j = 0; j < result.cols; j++){
      if((int)norm.at<float>(i, j) > threshold){
        key_points.push_back(KeyPoint(j, i, 2));
      }
    }
  }

  drawKeypoints(result, key_points, result);
  return result;
}

Mat orbHarris(Mat image){
  Mat result = image.clone();
  Mat gray, orb;
  Ptr<ORB> detector = ORB::create();
  vector<KeyPoint> key_points;
  Mat descriptors;

  cvtColor(result, gray, COLOR_BGR2GRAY);

  detector->detectAndCompute(gray, orb, key_points, descriptors);

  drawKeypoints(result, key_points, result);
  return result;
}

Mat orbFAST(Mat image){
  Mat result = image.clone();
  Mat gray, orb;
  Ptr<ORB> detector = ORB::create(500, 1.2F, 8, 31, 0, 2, ORB::FAST_SCORE, 31, 20);
  vector<KeyPoint> key_points;
  Mat descriptors;

  cvtColor(result, gray, COLOR_BGR2GRAY);

  detector->detectAndCompute(gray, orb, key_points, descriptors);

  drawKeypoints(result, key_points, result);
  return result;
}

Mat sift(Mat image){
  Mat result = image.clone();
  Mat gray, orb;
  Ptr<SIFT> detector = SIFT::create();
  vector<KeyPoint> key_points;
  Mat descriptors;

  cvtColor(result, gray, COLOR_BGR2GRAY);

  detector->detectAndCompute(gray, orb, key_points, descriptors);

  drawKeypoints(result, key_points, result);
  return result;
}

Mat surf(Mat image){
  Mat result = image.clone();
  Mat gray, orb;
  Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create();
  vector<KeyPoint> key_points;

  cvtColor(result, gray, COLOR_BGR2GRAY);
  
  surf->detect(gray, key_points);

  drawKeypoints(result, key_points, result);
  return result;
}

Mat akaze(Mat image){
  Mat result = image.clone();
  Mat gray, orb;
  Ptr<AKAZE> akaze = AKAZE::create();
  vector<KeyPoint> key_points;

  cvtColor(result, gray, COLOR_BGR2GRAY);

  akaze->detect(gray, key_points);

  drawKeypoints(result, key_points, result);
  return result;
}