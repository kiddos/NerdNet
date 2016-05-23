#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include "nnet.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define TOBIN(n) static_cast<unsigned char>(n < 0 ? 0 : n)
#define CLIP(n) n > 255 ? 255 : n;

using std::vector;
using std::string;
using std::cout;
using std::endl;
using nn::mat;
using nn::Layer;
using nn::InputLayer;
using nn::KullbackLeiblerOutput;
using nn::QuadraticOutput;
using nn::NeuralNet;
using nn::MomentumTrainer;

const string IMAGE_PATH = "/home/joseph/Pictures/bear1.png";
const int IMAGE_WIDTH = 240;
const int IMAGE_HEIGHT = 135;
const double NORM = 255;
const double thresh = 1e128;

double relu(double z) {
  if (z > thresh) return thresh;
  else if (z >= 0 && z <= thresh) return z;
  return 0;
}

double relud(double z) {
  if (z > thresh) return 0;
  else if (z >= 0 && z <= thresh) return 1;
  return 0;
}

mat cost(mat y, mat h) {
  const mat diff = y-h;
  return diff % diff / 2;
}

mat costd(mat y, mat a, mat z) {
  return (a - y) % nn::funcop(z, nn::sigmoid.actd);
}

void load(mat &x, mat &y) {
  cv::Mat image = cv::imread(IMAGE_PATH, CV_LOAD_IMAGE_COLOR);
  cv::resize(image, image, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
  x = mat(image.rows*image.cols, 2);
  y = mat(image.rows*image.cols, 3);
  y.zeros();
  for (int i = 0 ; i < image.rows ; ++i) {
    for (int j = 0 ; j < image.cols ; ++j) {
      x(i * image.cols + j, 0) = 2.0 * j / IMAGE_WIDTH - 1;
      x(i * image.cols + j, 1) = 2.0 * i / IMAGE_HEIGHT - 1;
      //x(i * image.cols + j, 0) = j;
      //x(i * image.cols + j, 1) = i;
      cv::Vec3b colors = image.at<cv::Vec3b>(i, j);
      y(i * image.cols + j, 0) = colors.val[0] / NORM;
      y(i * image.cols + j, 1) = colors.val[1] / NORM;
      y(i * image.cols + j, 2) = colors.val[2] / NORM;
    }
  }
}

double draw(cv::Mat& image, NeuralNet& nnet, const mat& y) {
  double cost = 0;
  for (int i = 0 ; i < image.rows ; ++i) {
    for (int j = 0 ; j < image.cols ; ++j) {
      mat x = mat(1, 2);
      x(0, 0) = 2.0 * j / IMAGE_WIDTH - 1;
      x(0, 1) = 2.0 * i / IMAGE_HEIGHT - 1;
      //x(0, 0) = j;
      //x(0, 1) = i;
      nnet.predict(x);
      const mat result = nnet.getresult();
      image.at<cv::Vec3b>(i, j)(0) = CLIP(TOBIN(result(0, 0) * NORM));
      image.at<cv::Vec3b>(i, j)(1) = CLIP(TOBIN(result(0, 1) * NORM));
      image.at<cv::Vec3b>(i, j)(2) = CLIP(TOBIN(result(0, 2) * NORM));
      cost += nnet.computecost(result, y.row(i*image.cols+j));
    }
  }
  return cost;
}

int main() {
  const double lrate = 1e-2;
  const double lambda = 0;
  const int batchsize = 10;
  const int n = 2;
  const int o = 3;

  srand(time(NULL));

  const int hsize = 20;
  InputLayer input(n);
  vector<Layer> hidden = {
    Layer(n, hsize, lrate, lambda, nn::relu),
    Layer(hsize, hsize, lrate, lambda, nn::relu),
    Layer(hsize, hsize, lrate, lambda, nn::sigmoid),
    Layer(hsize, hsize, lrate, lambda, nn::relu),
    Layer(hsize, hsize, lrate, lambda, nn::sigmoid),
    Layer(hsize, hsize, lrate, lambda, nn::sigmoid),
  };
  nn::OutputLayer output(hsize, o, lrate, lambda, nn::sigmoid, cost, costd);
  NeuralNet nnet(input, output, hidden);
  nn::MomentumTrainer trainer(nnet, 0.9);

  mat x, y;
  load(x, y);
  cout << y.row(y.n_rows-1) << endl;

  cv::namedWindow("Real Image");
  cv::namedWindow("NNet Image");
  cv::Mat realimage = cv::imread(IMAGE_PATH, CV_LOAD_IMAGE_COLOR);
  cv::resize(realimage, realimage, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
  cv::Mat nnetimage = cv::Mat::zeros(realimage.rows, realimage.cols, CV_8UC3);
  cv::imshow("Real Image", realimage);

  cout << "training start..." << endl;
  for (uint32_t i = 0 ; i < x.n_rows * 100 ; ++i) {
  //for (uint32_t i = 0 ; i < 3 ; ++i) {
    const int start = i % (x.n_rows-batchsize);
    const int end = start + batchsize;
    trainer.feeddata(x.rows(start, end), y.rows(start, end));
    //nnet.feeddata(x.row(i % x.n_rows), y.row(i % y.n_rows), false);

    //cout << nnet.gethidden(0).getgrad() << endl;
    //cout << nnet.gethidden(0).getw() << endl;
    //if (i % 10 == 0)
      //cout << "\riteration: " << i << " cost: " << nnet.computecost();
    if (i % x.n_rows == 0) {
      double newcost = draw(nnetimage, nnet, y);
      cout << endl << endl << "iteration: " << i << " cost: " << newcost << endl;
      //cout << y.rows(start, end) << endl;
      cout << nnet.getresult() << endl;
      cv::imshow("NNet Image", nnetimage);
      cv::waitKey(100);
    }
  }
  cout << endl << "training completed" << endl;
  cv::waitKey(0);

  return 0;
}
