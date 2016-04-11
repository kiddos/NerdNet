#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include "../src/nnet.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using std::vector;
using std::cout;
using std::endl;
using nn::Layer;
using nn::InputLayer;
using nn::OutputLayer;
using nn::NeuralNet;
using nn::mat;

double sigmoid(double z) {
  return 1.0 / (1.0 + exp(-z));
}

double sigmoidgrad(double z) {
  const double e = exp(-z);
  const double b = 1 + e;
  return e / (b * b);
}

mat cost(mat y, mat h) {
  mat J = -(y % nn::funcop(h, log) + (1-y) % nn::funcop(1-h, log));
  return J;
}

mat costd(mat y, mat a, mat z) {
  mat grad = (a - y);
  return grad;
}

void load(mat &x, mat &y) {
  std::ifstream xinput("/home/joseph/C/basic/src/nnet/samplex2.data", std::ios::in);
  std::ifstream yinput("/home/joseph/C/basic/src/nnet/sampley2.data", std::ios::in);
  x = mat(100, 2);
  y = mat(100, 2);
  y.zeros();
  if (xinput.is_open() && yinput.is_open()) {
    cout << "reading data..." << endl;
    for (uint32_t i = 0 ; i < 100 ; i ++) {
      double x1 = 0, x2 = 0;
      int yi = 0;
      xinput >> x1;
      xinput >> x2;
      yinput >> yi;
      x(i, 0) = x1;
      x(i, 1) = x2;
      y(i, yi) = 1;
    }
  }
}
void loadsample(mat &sample, const int w, const int h) {
  sample = mat(w * h, 2);
  for (int i = 0 ; i < h * w ; ++i) {
    sample(i, 0) = i / w;
    sample(i, 1) = i % w;
  }
}

int main() {
  const double lrate = 1e-3;
  const int w = 800;
  const int h = 600;

  InputLayer input(2);
  vector<Layer> hidden = {
    Layer(2, 11, lrate, sigmoid, sigmoidgrad),
    Layer(11, 15, lrate, sigmoid, sigmoidgrad),
    Layer(15, 21, lrate, sigmoid, sigmoidgrad),
    Layer(21, 3, lrate, sigmoid, sigmoidgrad),
    Layer(3, 5, lrate, sigmoid, sigmoidgrad),
    Layer(5, 10, lrate, sigmoid, sigmoidgrad)
  };
  OutputLayer output(10, 2, lrate, sigmoid, sigmoidgrad, cost, costd);
  NeuralNet nnet(input, output, hidden);

  mat x, y, sample;
  load(x, y); loadsample(sample, w, h);
  for (int i = 0 ; i < 30000 ; ++i) {
    nnet.feeddata(x, y);
    cout << "\rcost: " << nnet.computecost();
  }
  cout << endl;
  mat result = nnet.predict(sample);

  cv::Mat canvas = cv::Mat::zeros(h, w, CV_32FC1);
  cv::imshow("demo", canvas);

  return 0;
}
