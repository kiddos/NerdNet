#include <iostream>
#include <string>
#include <omp.h>
#include <time.h>
#include <cstdlib>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "nnet.h"

using std::string;
using std::endl;
using std::cout;
using nn::Conv2DLayer;
using nn::mat;

void resizeimg(mat& sample, cv::Mat img) {
  const int imgsize = img.rows * img.cols;
  sample = mat(1, img.rows*img.cols*img.channels());

  for (int c = 0 ; c < img.channels() ; ++c) {
    for (int i = 0 ; i < img.rows ; ++i) {
      for (int j = 0 ; j < img.cols ; ++j) {
        sample(c * imgsize + i * img.cols + j) = img.at<cv::Vec3b>(i, j)[c];
      }
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc != 9) return -1;
  string imgpath = argv[1];
  int w = atoi(argv[2]);
  int h = atoi(argv[3]);
  int nfilter = atoi(argv[4]);
  int spatial = atoi(argv[5]);
  int stride = atoi(argv[6]);
  int padding = atoi(argv[7]);
  double lrate = atof(argv[8]);

  //srand(time(nullptr));

  cv::Mat img = cv::imread(imgpath, CV_LOAD_IMAGE_COLOR);
  if (img.empty()) {
    cout << "Fail to load img" << endl;
    return -1;
  }

  cv::resize(img, img, cv::Size(w, h));

  Conv2DLayer layer(w, h, img.channels(), nfilter, spatial, stride, padding,
                    lrate, nn::relu);

  mat sample(0, 0);
  resizeimg(sample, img);
  double start = omp_get_wtime();
  mat out, delta;
  for (int i = 0 ; i < 1000 ; ++i) {
    out = layer.forwardprop(sample);
  }
  double pass = omp_get_wtime() - start;
  cout << "time used: " << pass << endl;;

  start = omp_get_wtime();
  for (int i = 0 ; i < 1000 ; ++i) {
    delta = layer.backprop(out);
  }
  pass = omp_get_wtime() - start;
  cout << "time used: " << pass << endl;;

  start = omp_get_wtime();
  for (int i = 0 ; i < 1000 ; ++i) {
    layer.forwardprop(sample);
    layer.backprop(out);
  }
  pass = omp_get_wtime() - start;
  cout << "time used: " << pass << endl;;

  //cout << "partial output image: " << endl;
  //mat outimage = out;
  //outimage.reshape(w, outimage.n_cols/w);
  //cout << outimage.n_rows << ", " << outimage.n_cols << endl;
  //for (uint32_t i = 0 ; i*h < outimage.n_cols ; ++i) {
    //cout << outimage.submat(0, i*h, w-1, (i+1)*h-1).t() << endl;
  //}

  //cout << "partial delta" << endl;
  //cout << delta << endl;
  //delta.reshape(img.channels(), delta.n_cols / img.channels());
  //for (uint32_t i = 0 ; i < delta.n_rows ; ++i) {
    //mat image = delta.row(i);
    //image.reshape(w, h);
    //image = image.t();
    //cout << image << endl;
  //}

  return 0;
}
