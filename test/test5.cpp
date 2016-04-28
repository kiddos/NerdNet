#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/shape.hpp>

#include "../src/nnet.h"

using std::vector;
using std::cout;
using std::endl;
using nn::Layer;
using nn::InputLayer;
using nn::OutputLayer;
using nn::NeuralNet;
using nn::mat;

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
  //mat J = -(y % h.transform(log) + (1-y) % nn::funcop(1-h, log)) / y.n_rows;
  //mat J = -(y % h);
  return J;
}

mat costd(mat y, mat a, mat,mat) {
  mat grad = (a - y);
  //mat grad = (a - y) / y.n_rows;
  //mat grad = -(y % nn::funcop(z, sigmoidgrad));
  return grad;
}

void load(mat &x, mat &y) {
  std::ifstream input("/home/joseph/C/data/iris.dat", std::ios::in);
  std::string line;
  const int datasize = 150;

  if (input.is_open()) {
    x = mat(datasize, 4);
    y = mat(datasize, 3);
    y.zeros();

    for (; std::getline(input, line) ;) {
      if (strcmp(line.substr(0, 5).c_str(), "@data") == 0) {
        cout << "reading data..." << endl;
        break;
      }
    }

    for (uint32_t i = 0 ; i < datasize ; ++i) {
      for (int j = 0 ; j < 4 ; ++j) {
        double xi = 0;
        char c;
        input >> xi;
        x(i, j) = xi;

        input >> c;
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

double accuracy(mat answer, mat prediction) {
  double correct = 0;
  for (uint32_t i = 0 ; i < answer.n_rows ; ++i) {
    const int index = static_cast<int>(prediction(i, 0));
    if (answer(i, index) == 1) {
      correct ++;
    }
  }
  return correct / static_cast<double>(answer.n_rows);
}

int main() {
  double lrate = 1e-3;
  const double lratedecay = 0.90;
  const double lambda = 1e-8;

  srand(time(NULL));

  InputLayer input(4);
  vector<Layer> hidden = {
    Layer(4, 3, lrate, lambda, sigmoid, sigmoidgrad),
    Layer(3, 3, lrate, lambda, sigmoid, sigmoidgrad),
    Layer(3, 6, lrate, lambda, sigmoid, sigmoidgrad),
  };
  OutputLayer output(6, 3, lrate, 0, sigmoid, sigmoidgrad, cost, costd);
  NeuralNet nnet(input, output, hidden);

  mat x, y;
  load(x, y);
  //cout << x << endl;
  //cout << y << endl;

  nnet.feeddata(x, y, true);
  for (uint32_t i = 0 ; i < x.n_rows * 2000 ; ++i) {
    nnet.feeddata(x, y, false);
    //nnet.feeddata(x.row(i%x.n_rows), y.row(i%x.n_rows), false);
    const double newcost = nnet.computecost();
    cout << "\riteration: " << i+1 << " cost: " << newcost;

    if (i % (x.n_rows * 10) == 0) {
      cout << endl << "iteration: " << i+1 << " cost: " << newcost;
    }

    if (newcost < 150 && i % (x.n_rows * 10) == 0) {
      if (newcost < 50) {
        lrate *= 0.96;
      } else {
        lrate *= lratedecay;
      }
      nnet.setlrate(lrate);
    }
  }
  cout << endl;
  mat result = nnet.predict(x);
  cout << "accuracy: " << accuracy(y, result) * 100 << endl;

  return 0;
}
