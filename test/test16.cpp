#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string>

#include <mgl2/qt.h>
#include <mgl2/mgl.h>

#include "../src/nnet.h"

using std::vector;
using std::cout;
using std::endl;
using nn::Layer;
using nn::InputLayer;
using nn::OutputLayer;
using nn::NeuralNet;
using nn::mat;

const int K = 3;
const int datasize = 304;
const int n = K;
const int o = 1;
const double scale = 100;
//const double tou = 1e3;

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
  //const mat diff = h - y;
  //const mat squared = (diff % diff) / tou;
  //const mat J = tou * nn::funcop(squared, exp);

  const mat diff = y - h;
  const mat J = (diff % diff) / 2.0;
  //mat J = -(y % h.transform(log) + (1-y) % nn::funcop(1-h, log));
  //mat J = -(y % h.transform(log) + (1-y) % nn::funcop(1-h, log)) / y.n_rows;
  //mat J = -(y % h);
  return J;
}

mat costd(mat y, mat a, mat) {
  //const mat diff = a - y;
  //const mat squared = (diff % diff) / tou;
  //const mat J = tou * nn::funcop(squared, exp);
  //const mat grad = (2.0/tou) * diff % J;
  //const mat grad = (a - y);
  mat grad = (a - y);
  //mat grad = (a - y) / y.n_rows;
  //mat grad = -y;
  return grad;
}

void load(mat &x, mat &y, const int k) {
  std::ifstream input("/home/joseph/C/data/exchange-rate-twi-may-1970-aug-1.csv",
        std::ios::in);
  std::string line;

  if (input.is_open()) {
    mat data(datasize, 1);
    x = mat(datasize-k-1, k);
    y = mat(datasize-k-1, 1);
    y.zeros();

    std::getline(input, line);

    for (uint32_t i = 0 ; i < datasize ; ++i) {
      double value = 0;
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

    // random the input data
    //cout << "shuffle data..." << endl;
    //for (uint32_t i = 0 ; i < x.n_rows ; ++i) {
      //int index1 = rand() % x.n_rows;
      //int index2 = rand() % x.n_rows;
      //mat xrow1 = x.row(index1);
      //mat xrow2 = x.row(index2);
      //mat yrow1 = y.row(index1);
      //mat yrow2 = y.row(index2);
      //for (uint32_t j = 0 ; j < x.n_cols ; ++j) {
        //x(index1, j) = xrow2(0, j);
        //x(index2, j) = xrow1(0, j);
      //}

      //for (uint32_t j = 0 ; j < y.n_cols ; ++j) {
        //y(index1, j) = yrow2(0, j);
        //y(index2, j) = yrow1(0, j);
      //}
    //}
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

mglData prediction;
mglData answer;
int sample(mglGraph* graph) {
  graph->AddLegend("prediction", "r");
  graph->AddLegend("answer", "b");
  graph->SetRanges(1970, 1995, 0, 120);
  graph->Axis("y");
  graph->Axis("x");
  graph->Label('y', "exchange rate");
  graph->Label('x', "time");
  graph->Plot(answer, "-b5");
  graph->Plot(prediction, "-r5");
  return 0;
}

int main() {
  double lrate = 1e-2;
  const double lratedecay = 0.96;
  const double lambda = 1e-6;

  srand(time(NULL));

  InputLayer input(n);
  vector<Layer> hidden = {
    Layer(n, 256, lrate, lambda, sigmoid, sigmoidgrad),
    Layer(256, 16, lrate, lambda, sigmoid, sigmoidgrad),
    Layer(16, 16, lrate, lambda, sigmoid, sigmoidgrad),
    Layer(16, 16, lrate, lambda, sigmoid, sigmoidgrad),
    Layer(16, 16, lrate, lambda, sigmoid, sigmoidgrad),
    Layer(16, 16, lrate, lambda, sigmoid, sigmoidgrad),
    Layer(16, 16, lrate, lambda, sigmoid, sigmoidgrad),
    Layer(16, 16, lrate, lambda, sigmoid, sigmoidgrad),
    Layer(16, 16, lrate, lambda, sigmoid, sigmoidgrad),
    Layer(16, 16, lrate, lambda, sigmoid, sigmoidgrad),
    Layer(16, 16, lrate, lambda, sigmoid, sigmoidgrad),
    Layer(16, 16, lrate, lambda, sigmoid, sigmoidgrad),
    Layer(16, 16, lrate, lambda, sigmoid, sigmoidgrad),
    Layer(16, 16, lrate, lambda, sigmoid, sigmoidgrad),
  };
  OutputLayer output(16, o, lrate, lambda, identity, identitygrad, cost, costd);
  NeuralNet nnet(input, output, hidden);

  mat x, y;
  load(x, y, K);
  cout << x.n_rows << endl;
  cout << y.n_rows << endl;
  //const int batchsize = x.n_rows / 30;

  //nnet.feeddata(x.row(1), y.row(1), true);
  double cost = DBL_MAX;
  for (uint32_t i = 0 ; i < x.n_rows * 1000 ; ++i) {
    //const int start = i % (x.n_rows-batchsize);
    //const int end = start + batchsize;
    //nnet.feeddata(x.rows(start, end), y.rows(start, end), false);
    nnet.feeddata(x.row(i%x.n_rows), y.row(i%x.n_rows), false);
    //nnet.feeddata(x, y, false);
    cout << "\riteration: " << i+1 << " cost: " << nnet.computecost()
          << "       ";
    if (i % (x.n_rows*10) == 0) {
      double costtemp = nnet.computecost();
      if (costtemp > cost || costtemp < 1e-7) break;

      cost = costtemp;
      cout << endl << "iteration: " << i+1 << " cost: " << cost
          << "     ";
      //cout << endl << nnet.getresult() << endl;
      //cout << y.row(i% datasize) << endl;
    }
    if (i % (x.n_rows*10) == 0) {
      lrate *= lratedecay;
      nnet.setlrate(lrate);
    }
  }
  cout << endl;
  mat result = nnet.predict(x);
  //for (uint32_t i = 0 ; i < y.n_rows ; ++i) {
    //cout << "predict: " << result(i, 1) * scale <<
        //" | answer: " << y(i, 0) * scale << endl;
  //}

  prediction = mglData(result.n_rows);
  answer = mglData(y.n_rows);
  for (uint32_t i = 0 ; i < y.n_rows ; ++i) {
    prediction.a[i] = result(i, 1) * scale;
    answer.a[i] = y(i, 0) * scale;
  }
  mglQT display(sample, "monthly total number of pigs slaughter");

  return display.Run();
}
