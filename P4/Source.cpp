#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace cv::detail;

using std::size;
using std::cout;
using std::endl;
using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;
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
const string HORIZONTAL_BUILDING[5] = { "72.JPG", "71.JPG", "73.JPG", "70.JPG", "74.JPG" };
const string SOMETHING[10] = { "800.jpeg", "801.jpeg", "802.jpeg", "803.jpeg", "804.jpeg", "805.jpeg",
    "806.jpeg", "807.jpeg", "808.jpeg", "809.jpeg" };

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
void makePanorama(const Mat& img1, const Mat& img2, Mat& img_panorama, float reject_ratio);


int main(){
  // Load and show images
  string window_name;
  string fich_name;
  Mat images[size(IMAGE_SET)];
  for (int i = 0; i < size(IMAGE_SET); i++){
    fich_name = PATH + IMAGE_SET[i];
    images[i] = imread(samples::findFile(fich_name), IMREAD_COLOR);
    if(images[i].empty()){
      printf("Error opening image: %s\n", fich_name.c_str());
      return EXIT_FAILURE;
    }
    imshow(to_string(i), images[i]);
  }
  waitKey(0);
  destroyAllWindows();

  // Make panorama
  float reject_ratio = 0.6;
  Mat panorama, aux;
  steady_clock::time_point begin = steady_clock::now();
  makePanorama(images[1], images[0], panorama, reject_ratio);
  steady_clock::time_point end = steady_clock::now();
  cout << "Time difference (sec) = " << (duration_cast<microseconds>(end - begin).count()) / 1000000.0 << endl;
  aux = panorama.clone();
  if ((panorama.rows > 1000) || (panorama.cols > 1900)){
    resize(aux, aux, Size(1900, 1000));
  }
  window_name = "Pan " + to_string(0) + "-" + to_string(1);
  imshow(window_name, aux);
  moveWindow(window_name, 10, 10);
  waitKey(0);
  destroyAllWindows();
  for (int i = 2; i < size(IMAGE_SET); i++){ // Add more images to the panorama
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    makePanorama(images[i], panorama, panorama, reject_ratio);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now(); 
    cout << "Time difference (sec) = " << (duration_cast<microseconds>(end - begin).count()) / 1000000.0 << endl;
    aux = panorama.clone();
    if ((panorama.rows > 1000) || (panorama.cols > 1900)){
      resize(aux, aux, Size(1900, 1000));
    }
    window_name = "Pan " + to_string(0) + "-" + to_string(i);
    imshow(window_name, aux);
    moveWindow(window_name, 10, 10);
    waitKey(0);
    destroyAllWindows();
  }

  return EXIT_SUCCESS;
}

void makePanorama(const Mat& img1, const Mat& img2, Mat& img_panorama, float reject_ratio){
  Mat img1_gray, img2_gray;
  cvtColor(img1, img1_gray, COLOR_BGR2GRAY);
  cvtColor(img2, img2_gray, COLOR_BGR2GRAY);

  // Find matches
  Mat desc1, desc2;
  vector<KeyPoint> kpts1, kpts2;
  vector<KeyPoint> matched1, matched2;
  vector<int> matchIdx1, matchIdx2;
  Ptr<SURF> surf = SURF::create();
  surf->detectAndCompute(img1_gray, noArray(), kpts1, desc1);
  surf->detectAndCompute(img2_gray, noArray(), kpts2, desc2);
  Ptr<DescriptorMatcher> matcher = BFMatcher::create(NormTypes::NORM_L2);

  // Search the matches but take only the best 2 of each descriptor
  vector<vector<DMatch>> nn_matches;
  matcher->knnMatch(desc1, desc2, nn_matches, 2);

  // Discard the matches too similar to the second best one
  matched1.clear();
  matched2.clear();
  for (size_t i = 0; i < nn_matches.size(); i++){
    DMatch first = nn_matches[i][0];
    float dist1 = nn_matches[i][0].distance;
    float dist2 = nn_matches[i][1].distance;
    if (dist1 < reject_ratio * dist2){
      matched1.push_back(kpts1[first.queryIdx]);
      matched2.push_back(kpts2[first.trainIdx]);
      matchIdx1.push_back(first.queryIdx);
      matchIdx2.push_back(first.trainIdx);
    }
  }

  // Show keypoints and matches
  Mat img_kpts;
  drawKeypoints(img1, kpts1, img_kpts);
  imshow("Keypoints 1", img_kpts);
  drawKeypoints(img2, kpts2, img_kpts);
  imshow("Keypoints 2", img_kpts);

  Mat img_matches;
  vector<DMatch> matches(matched1.size());
  for(int i = 0; i < matched1.size(); ++i){
    matches[i] = DMatch(i, i, 0);
  }
  drawMatches(img1, matched1, img2, matched2, matches, img_matches);
  imshow("Matches", img_matches);

  // RANSAC
  vector<Point2f> puntos_1, puntos_2;
  for (int i = 0; i < matched1.size(); i++){
      puntos_1.push_back(matched1[i].pt);
      puntos_2.push_back(matched2[i].pt);
  }
  vector<uchar> match_mask;
  Mat homography = findHomography(puntos_1, puntos_2, RANSAC, 3, match_mask, 2000, 0.995);
  
  // Count the inliers found in findHomography
  int inliers = 0;
  for (int i = 0; i < match_mask.size(); i++){
    if (match_mask[i] == 1){
      inliers++;
    }
  }
  cout << "Inliers found: " << inliers << endl;

  // Insert img2 on img1 using the homography
  vector<Point2f> corners_img1(4);
  corners_img1[0] = Point2f(0, 0);
  corners_img1[1] = Point2f(0, img1.rows);
  corners_img1[2] = Point2f(img1.cols, img1.rows);
  corners_img1[3] = Point2f(img1.cols, 0);

  vector<Point2f> corners_img2(4);
  corners_img2[0] = Point2f(0, 0);
  corners_img2[1] = Point2f(0, img2.rows);
  corners_img2[2] = Point2f(img2.cols, img2.rows);
  corners_img2[3] = Point2f(img2.cols, 0);

  vector<Point2f> corners_img1_warped;
  perspectiveTransform(corners_img1, corners_img1_warped, homography);
  vector<Point2f> all_corners = corners_img1_warped;
  all_corners.insert(all_corners.end(), corners_img2.begin(), corners_img2.end());

  // Calculate final size
  Point2f corner = all_corners[0];
  int x_min = corner.x;
  int y_min = corner.y;
  int x_max = x_min;
  int y_max = y_min;
  for (int i = 1; i < all_corners.size(); i++){
    corner = all_corners[i];
    x_min = min((int)corner.x, x_min);
    x_max = max((int)corner.x, x_max);
    y_min = min((int)corner.y, y_min);
    y_max = max((int)corner.y, y_max);
  }

  // Create translation matrix
  Mat translation = Mat::eye(3, 3, CV_64F);
  translation.at<double>(0, 2) = -x_min;
  translation.at<double>(1, 2) = -y_min;

  vector<Point2f> corners_result1;
  vector<Point2f> corners_result2;
  Size result_size = Size(x_max - x_min, y_max - y_min);
  Mat result1, result2;
  warpPerspective(img1, result1, translation * homography, result_size);
  perspectiveTransform(corners_img1, corners_result1, translation * homography);
  warpPerspective(img2, result2, translation, result_size);
  perspectiveTransform(corners_img2, corners_result2, translation);

  Mat mask_result1, mask_result2;
  Mat result1_gray;
  Mat result2_gray;
  cvtColor(result1, result1_gray, COLOR_BGR2GRAY);
  cvtColor(result2, result2_gray, COLOR_BGR2GRAY);
  threshold(result1_gray, mask_result1, 0, 255, THRESH_BINARY);
  threshold(result2_gray, mask_result2, 0, 255, THRESH_BINARY);

  // Mix the images
  Mat result;
  Mat element = getStructuringElement(0, Size(5, 5), Point(2, 2));
  erode(mask_result2, mask_result2, element);
  result = result1.clone();
  result2.copyTo(result, mask_result2);

  img_panorama = result;
}

void orbHarris(Mat image1, Mat image2, float reject_ratio, float scale_factor){
  Mat clone1 = image1.clone();
  Mat clone2 = image2.clone();
  Mat gray1, gray2, orb_result, result;
  Ptr<ORB> orbHarris = ORB::create(500, scale_factor, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
  vector<KeyPoint> key_points1, key_points2;
  Mat descriptors1, descriptors2;
  string window_name = "Orb Harris con ratio " + to_string(reject_ratio);
  vector<vector<DMatch>> matches;
  vector<DMatch> filtered_matches;

  cvtColor(clone1, gray1, COLOR_BGR2GRAY);
  cvtColor(clone2, gray2, COLOR_BGR2GRAY);

  orbHarris->detectAndCompute(gray1, orb_result, key_points1, descriptors1);
  orbHarris->detectAndCompute(gray2, orb_result, key_points2, descriptors2);

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
  Mat gray1, gray2, orb_result, result;
  Ptr<ORB> orbFAST = ORB::create(500, scale_factor, 8, 31, 0, 2, ORB::FAST_SCORE, 31, 20);
  vector<KeyPoint> key_points1, key_points2;
  Mat descriptors1, descriptors2;
  string window_name = "Orb FAST con ratio " + to_string(reject_ratio);
  vector<vector<DMatch>> matches;
  vector<DMatch> filtered_matches;

  cvtColor(clone1, gray1, COLOR_BGR2GRAY);
  cvtColor(clone2, gray2, COLOR_BGR2GRAY);

  orbFAST->detectAndCompute(gray1, orb_result, key_points1, descriptors1);
  orbFAST->detectAndCompute(gray2, orb_result, key_points2, descriptors2);

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
  Mat gray1, gray2, sift_result, result;
  Ptr<SIFT> sift = SIFT::create();
  vector<KeyPoint> key_points1, key_points2;
  Mat descriptors1, descriptors2;
  string window_name = "Sift con ratio " + to_string(reject_ratio);
  vector<vector<DMatch>> matches;
  vector<DMatch> filtered_matches;

  cvtColor(clone1, gray1, COLOR_BGR2GRAY);
  cvtColor(clone2, gray2, COLOR_BGR2GRAY);
  sift->detectAndCompute(gray1, sift_result, key_points1, descriptors1);
  sift->detectAndCompute(gray2, sift_result, key_points2, descriptors2);

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
  Mat gray1, gray2, surf_result, result;
  Ptr<SURF> surf = SURF::create();
  vector<KeyPoint> key_points1, key_points2;
  Mat descriptors1, descriptors2;
  string window_name = "Surf con ratio " + to_string(reject_ratio);
  vector<vector<DMatch>> matches;
  vector<DMatch> filtered_matches;

  cvtColor(clone1, gray1, COLOR_BGR2GRAY);
  cvtColor(clone2, gray2, COLOR_BGR2GRAY);

  surf->detectAndCompute(gray1, surf_result, key_points1, descriptors1);
  surf->detectAndCompute(gray2, surf_result, key_points2, descriptors2);

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
  Mat gray1, gray2, akaze_result, result;
  Ptr<AKAZE> akaze = AKAZE::create();
  vector<KeyPoint> key_points1, key_points2;
  Mat descriptors1, descriptors2;
  string window_name = "Akaze con ratio " + to_string(reject_ratio);
  vector<vector<DMatch>> matches;
  vector<DMatch> filtered_matches;

  cvtColor(clone1, gray1, COLOR_BGR2GRAY);
  cvtColor(clone2, gray2, COLOR_BGR2GRAY);

  akaze->detectAndCompute(gray1, akaze_result, key_points1, descriptors1);
  akaze->detectAndCompute(gray2, akaze_result, key_points2, descriptors2);

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