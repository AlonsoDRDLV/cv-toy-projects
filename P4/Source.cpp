#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>

using namespace cv;
using namespace cv::detail;

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
const string HORIZONTAL_BUILDING[5] = { "72.JPG", "71.JPG", "73.JPG", "70.JPG", "74.JPG" };

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
vector<Point2f> getCorners(const Mat& img);
void doPanorama(const Mat& img1, const Mat& img2, Mat& img_panorama, float reject_ratio);
Mat warpImages(const Mat& img1, const Mat& img2, const Mat& homography);


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
  doPanorama(images[1], images[0], panorama, reject_ratio);
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
    doPanorama(images[i], panorama, panorama, reject_ratio);
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

void doPanorama(const Mat& img1, const Mat& img2, Mat& img_panorama, float reject_ratio){
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

  // Busca un máximo de 2 emparejamientos por descriptor
  vector<vector<DMatch>> nn_matches;
  matcher->knnMatch(desc1, desc2, nn_matches, 2);

  // Conservar emparejamiento si es mejor que el segundo más parecido
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

  // Mostrar keypoints y matches
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
  waitKey(0);

   // RANSAC
   vector<Point2f> puntos_1, puntos_2;
   for (int i = 0; i < matched1.size(); i++){
       puntos_1.push_back(matched1[i].pt);
       puntos_2.push_back(matched2[i].pt);
   }
   Mat homography = findHomography(puntos_1, puntos_2, RANSAC);

  img_panorama = warpImages(img1, img2, homography);
}

Mat warpImages(const Mat& img1, const Mat& img2, const Mat& homography){
  // Transformar img1 según homography y colocarle encima img2
  vector<Point2f> corners_img1 = getCorners(img1);
  vector<Point2f> corners_img2 = getCorners(img2);
  vector<Point2f> corners_img1_warped;
  perspectiveTransform(corners_img1, corners_img1_warped, homography);
  vector<Point2f> all_corners = corners_img1_warped;
  all_corners.insert(all_corners.end(), corners_img2.begin(), corners_img2.end());

  // Buscar las dimensiones de la imagen final
  int xmin = INT_MAX, xmax = INT_MIN;
  int ymin = INT_MAX, ymax = INT_MIN;
  for (auto corner : all_corners){
    xmin = min((int)corner.x, xmin);
    xmax = max((int)corner.x, xmax);
    ymin = min((int)corner.y, ymin);
    ymax = max((int)corner.y, ymax);
  }

  // Crear matriz de traslacion
  Mat translation = Mat::eye(3, 3, CV_64F);
  translation.at<double>(0, 2) = -xmin;
  translation.at<double>(1, 2) = -ymin;


  vector<Point2f> corners_result1;
  vector<Point2f> corners_result2;
  Size result_size = Size(xmax - xmin, ymax - ymin);
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

  // Mezclar las dos imágenes
  Mat result;
  int erosion_type = 0;
  int erosion_size = 2;
  Mat element = getStructuringElement(erosion_type,
      Size(2 * erosion_size + 1, 2 * erosion_size + 1),
      Point(erosion_size, erosion_size));
  erode(mask_result2, mask_result2, element);
  result = result1.clone();
  result2.copyTo(result, mask_result2);

  return result;
}

vector<Point2f> getCorners(const Mat& img){
  int cols = img.cols;
  int rows = img.rows;
  vector<Point2f> corners(4);
  corners[0] = Point2f(0, 0);
  corners[1] = Point2f(0, rows);
  corners[2] = Point2f(cols, rows);
  corners[3] = Point2f(cols, 0);
  return corners;
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