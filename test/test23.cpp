#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include "../src/nnet.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define TOBIN(n) static_cast<unsigned char>(n < 0 ? 0 : n)

using std::vector;
using std::string;
using std::cout;
using std::endl;
using nn::Layer;
using nn::InputLayer;
using nn::QuadraticOutput;
using nn::NeuralNet;
using nn::mat;

const string IMAGE_PATH = "/home/joseph/Pictures/bear1.png";
const int IMAGE_WIDTH = 240;
const int IMAGE_HEIGHT = 135;

void load(mat &x, mat &y) {
  cv::Mat image = cv::imread(IMAGE_PATH, CV_LOAD_IMAGE_COLOR);
  cv::resize(image, image, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
  x = mat(image.rows*image.cols, 2);
  y = mat(image.rows*image.cols, 3);
  for (int i = 0 ; i < image.rows ; ++i) {
    for (int j = 0 ; j < image.cols ; ++j) {
      x(i * image.cols + j, 0) = 2.0 * j / IMAGE_WIDTH - 1;
      x(i * image.cols + j, 1) = 2.0 * i / IMAGE_HEIGHT - 1;
      cv::Vec3b colors = image.at<cv::Vec3b>(i, j);
      y(i * image.cols + j, 0) = colors.val[0];
      y(i * image.cols + j, 1) = colors.val[1];
      y(i * image.cols + j, 2) = colors.val[2];
    }
  }
}

void draw(cv::Mat& image, NeuralNet& nnet) {
  for (int i = 0 ; i < image.rows ; ++i) {
    for (int j = 0 ; j < image.cols ; ++j) {
      mat x = mat(1, 2);
      x(0, 0) = 2.0 * j / IMAGE_WIDTH - 1;
      x(0, 1) = 2.0 * i / IMAGE_HEIGHT - 1;
      nnet.predict(x);
      const mat result = nnet.getresult();
      image.at<cv::Vec3b>(i, j)(0) = TOBIN(result(0, 0));
      image.at<cv::Vec3b>(i, j)(1) = TOBIN(result(0, 1));
      image.at<cv::Vec3b>(i, j)(2) = TOBIN(result(0, 2));
    }
  }
}

int main() {
  const double lrate = 5e-7;
  const double lambda = 0;
  const int batchsize = 0;
  const int n = 2;
  const int o = 3;

  srand(time(NULL));

  InputLayer input(n);
  vector<Layer> hidden = {
    Layer(n, 100, lrate, lambda, nn::relu),
  };
  QuadraticOutput output(100, o, lrate, lambda);
  NeuralNet nnet(input, output, hidden);

  mat x, y;
  load(x, y);
  cout << y.row(y.n_rows-1) << endl;
  //nnet.feeddata(x.row(0), y.row(0), true);

  cv::namedWindow("Real Image");
  cv::namedWindow("NNet Image");
  cv::Mat realimage = cv::imread(IMAGE_PATH, CV_LOAD_IMAGE_COLOR);
  cv::resize(realimage, realimage, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
  cv::Mat nnetimage = cv::Mat::zeros(realimage.rows, realimage.cols, CV_8UC3);
  cv::imshow("Real Image", realimage);
  cout << "training start..." << endl;
  for (uint32_t i = 0 ; i < x.n_rows * 100 ; ++i) {
    const int start = i % (x.n_rows-batchsize);
    const int end = start + batchsize;
    nnet.feeddata(x.rows(start, end), y.rows(start, end), false);
    //nnet.feeddata(x.row(i % x.n_rows), y.row(i % y.n_rows), false);

    if (i % 10 == 0)
      cout << "\riteration: " << i << " cost: " << nnet.computecost();
    if (i % x.n_rows == 0) {
      const double cost = nnet.computecost();
      if (cost < 0.01) break;
      cout << endl << "iteration: " << i << " cost: " << cost << endl;
      //cout << y.rows(start, end) << endl;
      draw(nnetimage, nnet);
      cout << nnet.getresult() << endl;
      cv::imshow("NNet Image", nnetimage);
      cv::waitKey(100);
    }
  }
  cout << endl << "training completed" << endl;
  cv::waitKey(0);

  return 0;
}
