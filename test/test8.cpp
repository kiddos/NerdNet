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

const int datasize = 5404;
const int n = 5;

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
  std::ifstream input("/home/joseph/C/data/phoneme.dat", std::ios::in);
  std::string line;

  if (input.is_open()) {
    x = mat(datasize, n);
    y = mat(datasize, 2);
    y.zeros();

    for (; std::getline(input, line) ;) {
      if (strcmp(line.substr(0, 5).c_str(), "@data") == 0) {
        cout << "reading data..." << endl;
        break;
      }
    }

    for (uint32_t i = 0 ; i < datasize ; ++i) {
      for (int j = 0 ; j < n ; ++j) {
        double xi = 0;
        char c;
        input >> xi;
        x(i, j) = xi;

        // read ,
        input >> c;
      }

      double val = 0;
      input >> val;
      y(i, val) = 1;
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
  double lrate = 1e-2;
  const double lratedecay = 0.6;
  const double lambda = 9e-1;
  const int batchsize = 100;

  srand(time(NULL));

  InputLayer input(5);
  vector<Layer> hidden = {
    Layer(5, 64, lrate, lambda, sigmoid, sigmoidgrad),
    Layer(64, 16, lrate, lambda, sigmoid, sigmoidgrad),
    Layer(16, 4, lrate, lambda, sigmoid, sigmoidgrad),
    //Layer(16, 8, lrate, lambda, atan, [](double x) {return 1.0/(1.0+x*x);}),
  };
  OutputLayer output(4, 2, lrate, lambda, sigmoid, sigmoidgrad, cost, costd);
  NeuralNet nnet(input, output, hidden);

  mat x, y;
  load(x, y);
  //cout << x << endl;
  //cout << y << endl;

  //nnet.feeddata(x, y, true);
  for (int i = 0 ; i < 160000 ; ++i) {
    nnet.feeddata(x.rows(i%(datasize-batchsize), i%(datasize-batchsize)+batchsize-1),
                  y.rows(i%(datasize-batchsize), i%(datasize-batchsize)+batchsize-1),
                  false);
    cout << "\riteration: " << i+1 << " cost: " << nnet.computecost();
    if (i % datasize == 0) {
      cout << endl << "iteration: " << i+1 << " cost: " << nnet.computecost();
    }
    if (i % 1000 == 0) {
      lrate *= lratedecay;
      nnet.setlrate(lrate);
    }
  }
  cout << endl;
  mat result = nnet.predict(x);
  cout << "accuracy: " << accuracy(y, result) * 100 << endl;

  return 0;
}
