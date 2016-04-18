#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "../src/nnet.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/shape.hpp>

using std::vector;
using std::cout;
using std::endl;
using nn::Layer;
using nn::InputLayer;
using nn::OutputLayer;
using nn::NeuralNet;
using nn::mat;

#define DATA_SIZE 42000

double rectifier(double z) {
  return z >= 0 ? z : 0;
}

double rectifiergrad(double z) {
  return z >= 0 ? 1 : 0;
}

double sigmoid(double z) {
  return 1.0 / (1.0 + exp(-z));
}

double sigmoidgrad(double z) {
  const double e = exp(-z);
  const double b = 1 + e;
  return e / (b * b);
}

mat cost(mat y, mat h) {
  mat J = -(y % h.transform(log) + (1-y) % nn::funcop(1-h, log));
  return J;
}

mat costd(mat y, mat a, mat) {
  mat grad = (a - y);
  return grad;
}

void load(mat &x, mat &y) {
  std::ifstream input("/home/joseph/Python/competition/digits/train.raw",
          std::ios::in);
  const int datasize = 42000;
  x = mat(datasize, 784);
  y = mat(datasize, 10);
  y.zeros();
  if (input.is_open()) {
    cout << "reading data..." << endl;
    for (uint32_t i = 0 ; i < datasize ; ++i) {
      int yi = 0;
      input >> yi;
      y(i, yi) = 1;
      for (int j = 0 ; j < 784 ; ++j) {
        int xi = 0;
        input >> xi;
        x(i, j) = xi;
      }
    }
    cout << "done loading data." << endl;
  }
}
void loadsample(mat &sample, const int w, const int h) {
  sample = mat(w * h, 2);
  for (int i = 0 ; i < h * w ; ++i) {
    sample(i, 0) = static_cast<double>(1.0 * (i % w) / w);
    sample(i, 1) = static_cast<double>(1.0 * (i / w) / h);
  }
}

int main() {
  const double lrate = 1e-3;
  const int w = 800;
  const int h = 600;

  srand(time(NULL));

  InputLayer input(784);
  vector<Layer> hidden = {
    Layer(784, 64, lrate, sigmoid, sigmoidgrad),
  };
  OutputLayer output(64, 10, lrate, sigmoid, sigmoidgrad, cost, costd);
  NeuralNet nnet(input, output, hidden);

  mat x, y, sample;
  load(x, y); loadsample(sample, w, h);
  cout << "training start..." << endl;

  for (int i = 0 ; i < DATA_SIZE * 3000 ; ++i) {
    nnet.feeddata(x.row(i % DATA_SIZE), y.row(i % DATA_SIZE), false);
    cout << "\riteration: " << i + 1 << " cost: " << nnet.computecost();
  }
  cout << endl << "training completed" << endl;
  mat result = nnet.predict(x);

  //cv::Mat canvas = cv::Mat::zeros(h, w, CV_8UC1);
  //for (uint32_t i = 0 ; i < result.n_rows ; ++i) {
    //canvas.at<uchar>(i/w, i%w) = result(i, 0) == 1 ? 128 : 0;
  //}
  //for (uint32_t i = 0 ; i < x.n_rows ; ++i) {
    //if (y(i,0) == 0) {
      //cv::circle(canvas, cv::Point(x(i,0)*w, x(i,1)*h), 3, cv::Scalar(255));
    //} else {
      //cv::circle(canvas, cv::Point(x(i,0)*w, x(i,1)*h), 3, cv::Scalar(100));
    //}
  //}

  //cv::imshow("demo", canvas);
  //cv::waitKey(0);

  return 0;
}
