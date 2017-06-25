#include <iostream>
#include <vector>
#include <fstream>
#include <memory>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "nnet.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/shape.hpp>

using std::vector;
using std::shared_ptr;
using std::cout;
using std::endl;
using nn::mat;
using nn::Layer;
using nn::InputLayer;
using nn::OutputLayer;
using nn::NeuralNet;
using nn::Trainer;
using cv::Scalar;
using cv::Vec3b;

void load(mat &x, mat &y) {
  std::ifstream xinput("./data/samplex3.data", std::ios::in);
  std::ifstream yinput("./data/sampley3.data", std::ios::in);
  x = mat(300, 2);
  y = mat(300, 3);
  y.zeros();
  if (xinput.is_open() && yinput.is_open()) {
    cout << "reading data..." << endl;
    for (uint32_t i = 0 ; i < 300 ; i ++) {
      double x0 = 0, x1 = 0;
      int yi = 0;
      xinput >> x0;
      xinput >> x1;
      yinput >> yi;

      x(i, 0) = x0;
      x(i, 1) = x1;
      y(i, yi) = 1;
    }
  }
}
void loadsample(mat &sample, const int w, const int h) {
  sample = mat(w * h, 2);
  for (int i = 0 ; i < h * w ; ++i) {
    //sample(i, 0) = static_cast<double>(1.0 * (i % (w/2)) / (w/2) - 1);
    //sample(i, 1) = static_cast<double>(1.0 * (i / (w/2)) / (h/2) - 1);
    sample(i, 0) = static_cast<double>(1.0 * (i % w) / w * 2 - 1);
    sample(i, 1) = static_cast<double>(1.0 * (i / w) / h * 2 - 1);
  }
}

int main() {
  const double lrate = 1e-2;
  const double lambda = 0;
  const int w = 800;
  const int h = 600;

  // set up model
  InputLayer input(2);
  vector<shared_ptr<Layer>> hidden = {
    std::make_shared<Layer>(2, 64, lrate, 1e-2, lambda, nn::arctan),
    std::make_shared<Layer>(64, 16, lrate, 1e-2, lambda, nn::arctan)
  };
  nn::SoftmaxOutput output(16, 3, lrate, lambda);
  NeuralNet nnet(input, output, hidden);
  Trainer trainer(nnet);

  // loading data
  mat x, y, sample;
  load(x, y); loadsample(sample, w, h);

  // gradient check
  // trainer.gradcheck(x.row(0), y.row(0));
  // training
  for (int i = 0 ; i < 36000 ; ++i) {
    const double cost = trainer.feeddata(x, y, true);
    cout << "\riteration: " << i << " | cost: " << cost;
  }
  cout << endl;

  // perform prediction
  mat result = nnet.predict(sample);

  // draw the prediction
  cv::Mat canvas = cv::Mat::zeros(h, w, CV_8UC3);
  for (uint32_t i = 0 ; i < result.n_rows ; ++i) {
    if (result(i,0) == 0) {
      canvas.at<Vec3b>(i/w, i%w) = Vec3b(255, 100, 100);
    } else if (result(i,0) == 1) {
      canvas.at<Vec3b>(i/w, i%w) = Vec3b(100, 255, 100);
    } else if (result(i,0) == 2) {
      canvas.at<Vec3b>(i/w, i%w) = Vec3b(100, 100, 255);
    }
  }

  // draw the sample
  cout << "drawing training sample..." << endl;
  for (uint32_t i = 0 ; i < x.n_rows ; ++i) {
    if (y(i,0) == 1) {
      cv::circle(canvas, cv::Point(x(i,0)*w/2+w/2, x(i,1)*h/2+h/2),
                 3, Scalar(160, 30, 30));
    } else if (y(i,1) == 1) {
      cv::circle(canvas, cv::Point(x(i,0)*w/2+w/2, x(i,1)*h/2+h/2),
                 3, Scalar(30, 160, 30));
    } else if (y(i,2) == 1) {
      cv::circle(canvas, cv::Point(x(i,0)*w/2+w/2, x(i,1)*h/2+h/2),
                 3, Scalar(30, 30, 160));
    }
  }

  cv::imshow("demo", canvas);
  cv::waitKey(0);

  return 0;
}
