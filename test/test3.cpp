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

const int datasize = 10;

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
  std::ifstream input("/home/joseph/C/project/nn/data/dataset1", std::ios::in);
  x = mat(datasize, 11);
  y = mat(datasize, 2);
  y.zeros();

  if (input.is_open()) {
    cout << "reading data..." << endl;
    for (uint32_t i = 0 ; i < datasize ; ++i) {
      for (int j = 0 ; j < 11 ; ++j) {
        int xi = 0;
        input >> xi;
        x(i, j) = xi;
      }
      int yi = 0;
      input >> yi;
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
  const int w = 800;
  const int h = 600;

  srand(time(NULL));

  InputLayer input(2);
  vector<Layer> hidden = {
    Layer(2, 3, lrate, atan, [](double x) {return 1.0/(1.0+x*x);}),
    Layer(3, 10, lrate, rectifier, rectifiergrad),
    Layer(10, 2, lrate, atan, [](double x) {return 1.0/(1.0+x*x);}),
    Layer(2, 15, lrate, sigmoid, sigmoidgrad),
  };
  OutputLayer output(15, 2, lrate, sigmoid, sigmoidgrad, cost, costd);
  NeuralNet nnet(input, output, hidden);

  mat x, y, sample;
  load(x, y); loadsample(sample, w, h);
  for (int i = 0 ; i < 30000 ; ++i) {
    nnet.feeddata(x, y, false);
    //nnet.feeddata(x, y, true);
    cout << "\riteration: " << i+1 << " cost: " << nnet.computecost();
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
