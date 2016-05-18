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
using nn::mat;
using nn::Layer;
using nn::InputLayer;
using nn::OutputLayer;
using nn::NeuralNet;
using nn::Trainer;

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
  mat J = -(y % nn::funcop(h, log) + (1-y) % nn::funcop(1-h, log));
  return J;
}

mat costd(mat y, mat a, mat) {
  mat grad = (a - y);
  return grad;
}

void load(mat &x, mat &y) {
  std::ifstream xinput("./data/samplex.data", std::ios::in);
  std::ifstream yinput("./data/sampley.data", std::ios::in);
  x = mat(100, 2);
  y = mat(100, 2);
  y.zeros();
  if (xinput.is_open() && yinput.is_open()) {
    cout << "reading data..." << endl;
    for (uint32_t i = 0 ; i < 100 ; i ++) {
      double x0 = 0, x1 = 0, x2 = 0;
      int yi = 0;
      xinput >> x0;
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
    sample(i, 0) = static_cast<double>(1.0 * (i % w) / w);
    sample(i, 1) = static_cast<double>(1.0 * (i / w) / h);
  }
}

int main() {
  const double lrate = 1e-3;
  const double lambda = 1e-6;
  const int w = 800;
  const int h = 600;

  srand(time(NULL));

  InputLayer input(2);
  vector<Layer> hidden = {
    //Layer(2, 2, lrate, rectifier, rectifiergrad),
    //Layer(2, 6, lrate, atan, [](double x) {return 1.0/(1.0+x*x);}),
    Layer(2, 6, lrate, lambda, atan, [](double x) {return 1.0/(1.0+x*x);}),
    Layer(6, 6, lrate, lambda, rectifier, rectifiergrad),
    Layer(6, 6, lrate, lambda, sigmoid, sigmoidgrad),
    Layer(6, 6, lrate, lambda, atan, [](double x) {return 1.0/(1.0+x*x);}),
    //Layer(2, 6, lrate, sigmoid, sigmoidgrad),
    //Layer(6, 6, lrate, sigmoid, sigmoidgrad),
    //Layer(6, 6, lrate, sigmoid, sigmoidgrad),
    //Layer(6, 6, lrate, sigmoid, sigmoidgrad),
  };
  OutputLayer output(6, 2, lrate, lambda, sigmoid, sigmoidgrad, cost, costd);
  NeuralNet nnet(input, output, hidden);
  Trainer trainer(nnet);

  mat x, y, sample;
  load(x, y); loadsample(sample, w, h);

  trainer.gradcheck(x, y);
  for (int i = 0 ; i < 120000 ; ++i) {
    trainer.feeddata(x, y);
    cout << "\riteration: " << i << " | cost: " << nnet.computecost();
  }
  cout << endl;
  mat result = nnet.predict(sample);

  cv::Mat canvas = cv::Mat::zeros(h, w, CV_8UC1);
  for (uint32_t i = 0 ; i < result.n_rows ; ++i) {
    canvas.at<uchar>(i/w, i%w) = result(i, 0) == 1 ? 128 : 0;
  }
  for (uint32_t i = 0 ; i < x.n_rows ; ++i) {
    if (y(i,0) == 0) {
      cv::circle(canvas, cv::Point(x(i,0)*w, x(i,1)*h), 3, cv::Scalar(255));
    } else {
      cv::circle(canvas, cv::Point(x(i,0)*w, x(i,1)*h), 3, cv::Scalar(100));
    }
  }

  cv::imshow("demo", canvas);
  cv::waitKey(0);

  return 0;
}
