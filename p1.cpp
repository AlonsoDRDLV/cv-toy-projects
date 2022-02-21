#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
// we're NOT "using namespace std;" here, to avoid collisions between the beta variable and std::beta in c++17
using std::cin;
using std::cout;
using std::endl;
using namespace cv;

void contrast_effect(Mat* image, double alpha){
  for (int y = 0; y < image->rows; y++) {
    for (int x = 0; x < image->cols; x++) {
      for (int c = 0; c < image->channels(); c++) {
        image->at<Vec3b>(y, x)[c] =
          saturate_cast<uchar>(alpha * image->at<Vec3b>(y, x)[c]);
      }
    }
  }
}

int main(int argc, char** argv)
{
  VideoCapture capture;
  capture.open(0);
  Mat image;

  double alpha = 1.0; /*< Simple contrast control */
  cout << " Basic Linear Transforms " << endl;
  cout << "-------------------------" << endl;
  cout << "* Enter the alpha value [1.0-3.0]: ";
  cin >> alpha;

  if (!capture.isOpened()) return -2;
  
  while (true)
  {
    capture.read(image);
    if (image.empty())
    {
      cout << "Could not open or find the image!\n" << endl;
      cout << "Usage: " << argv[0] << " <Input image>" << endl;
      return -1;
    }

    contrast_effect(&image, alpha);

    imshow("Original Image", image);

    int c = waitKey(10);
    if ((char)c == 'q')
      break;
  }
  return 0;
}