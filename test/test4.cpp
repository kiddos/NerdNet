#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "../src/nnet.h"
#include <string>

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

const int datasize = 11;

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
  std::ifstream input("./data/dataset2", std::ios::in);
  x = mat(datasize, 4);
  y = mat(datasize, 3);
  y.zeros();

  if (input.is_open()) {
    cout << "reading data..." << endl;
    for (uint32_t i = 0 ; i < datasize ; ++i) {
      for (int j = 0 ; j < 4 ; ++j) {
        double xi = 0;
        input >> xi;
        x(i, j) = xi;
      }
      std::string label;
      input >> label;
      if (strcmp(label.c_str(), "Iris-setosa") == 0)
        y(i, 0) = 1;
      else if (strcmp(label.c_str(), "Iris-versicolor") == 0)
        y(i, 1) = 1;
      else if (strcmp(label.c_str(), "Iris-virginica") == 0)
        y(i, 2) = 1;
    }
  }
}

int main() {
  const double lrate = 1e-3;
  const double lambda = 1e-5;

  srand(time(NULL));

  InputLayer input(4);
  vector<Layer> hidden = {
    Layer(4, 6, lrate, lambda, atan, [](double x) {return 1.0/(1.0+x*x);}),
  };
  OutputLayer output(6, 3, lrate, lambda, sigmoid, sigmoidgrad, cost, costd);
  NeuralNet nnet(input, output, hidden);
  Trainer trainer(nnet);

  mat x, y;
  load(x, y);

  trainer.gradcheck(x.row(0), y.row(0));
  for (int i = 0 ; i < 210000 ; ++i) {
    trainer.feeddata(x, y);
    cout << "\riteration: " << i+1 << " cost: " << nnet.computecost();
  }
  cout << endl;
  mat result = nnet.predict(x);

  for (uint32_t i = 0 ; i < x.n_rows ; ++i) {
    for (uint32_t j = 0 ; j < x.n_cols ; ++j) {
      cout << x(i, j) << " ";
    }
    cout << "prediction: ";
    if (result(i, 0) == 0) cout << "Iris-setosa";
    if (result(i, 0) == 1) cout << "Iris-vericolor";
    if (result(i, 0) == 2) cout << "Iris-virginica";
    cout << " answer: ";
    if (y(i, 0) == 1) cout << "Iris-setosa";
    if (y(i, 1) == 1) cout << "Iris-vericolor";
    if (y(i, 2) == 1) cout << "Iris-virginica";
    cout << endl;
  }

  return 0;
}
