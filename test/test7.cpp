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

const int K = 10;
const int datasize = 188;
const double scale = 5000;
const double tou = 1000;

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

double identity(double z) {
  return z;
}

double identitygrad(double) {
  return 1;
}

mat cost(mat y, mat h) {
  const mat diff = h - y;
  const mat squared = (diff % diff) / tou;
  mat J = tou * nn::funcop(squared, exp);

  //const mat diff = y - h;
  //const mat J = (diff % diff) / 2.0;
  //mat J = -(y % h.transform(log) + (1-y) % nn::funcop(1-h, log));
  //mat J = -(y % h.transform(log) + (1-y) % nn::funcop(1-h, log)) / y.n_rows;
  //mat J = -(y % h);
  return J;
}

mat costd(mat y, mat a, mat,mat) {
  const mat diff = a - y;
  const mat squared = (diff % diff) / tou;
  const mat J = tou * nn::funcop(squared, exp);
  const mat grad = (2.0/tou) * diff % J;
  //const mat grad = (a - y);
  //mat grad = (a - y);
  //mat grad = (a - y) / y.n_rows;
  //mat grad = -(y % nn::funcop(z, sigmoidgrad));
  return grad;
}

void load(mat &x, mat &y, const int k) {
  std::ifstream input("/home/joseph/C/data/monthly-total-number-of-pigs-sla.csv",
        std::ios::in);
  std::string line;

  if (input.is_open()) {
    mat data(datasize, 1);
    x = mat(datasize-k-1, k);
    y = mat(datasize-k-1, 1);
    y.zeros();

    std::getline(input, line);

    for (uint32_t i = 0 ; i < datasize ; ++i) {
      int value = 0;
      char c = '\0';
      for (int j = 0 ; j < 10 ; ++j) input >> c;
      input >> value;
      data(i, 0) = value / scale;
    }
    input.close();

    //cout << data << endl;
    for (int i = 0 ; i < datasize - k - 1; ++i) {
      for (int j = i ; j < i + k ; ++j) {
        x(i, j-i) = data(j, 0);
      }
      y(i, 0) = data(i + k);
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
  double lrate = 1e-5;
  const double lratedecay = 1e-9;
  const double lambda = 999e-3;

  srand(time(NULL));

  InputLayer input(K);
  vector<Layer> hidden = {
    Layer(K, K, lrate, lambda, atan, [](double x) {return 1.0 / (1.0+x*x);}),
  };
  OutputLayer output(K, 1, lrate, identity, identitygrad, cost, costd);
  NeuralNet nnet(input, output, hidden);

  mat x, y;
  load(x, y, K);
  //cout << x << endl;
  //cout << y << endl;

  //nnet.feeddata(x, y, true);
  for (int i = 0 ; i < 260000 ; ++i) {
    //nnet.feeddata(x.row(i % (datasize-K-1)), y.row(i % (datasize-K-1)), false);
    nnet.feeddata(x, y, false);
    if (i > 80000) {
      nnet.setlrate(lrate);
      if (lrate > 1e-7)
        lrate -= lratedecay;
    }
    cout << "\riteration: " << i+1 << " cost: " << nnet.computecost();
  }
  cout << endl;
  mat result = nnet.predict(x);
  for (uint32_t i = 0 ; i < y.n_rows ; ++i) {
    cout << "predict: " << result(i, 1) * scale <<
        " | answer: " << y(i, 0) * scale << endl;
  }

  return 0;
}
