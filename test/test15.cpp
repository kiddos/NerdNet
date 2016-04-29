#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string>

#include <mgl2/mgl.h>
#include <mgl2/qt.h>

#include "../src/nnet.h"

using std::vector;
using std::cout;
using std::endl;
using nn::Layer;
using nn::InputLayer;
using nn::OutputLayer;
using nn::NeuralNet;
using nn::mat;

const int datasize = 22784;
const int n = 16;
const int o = 1;
const double scale = 500001.0 / 50.0;

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


mat colsum(const mat m) {
  mat result(m.n_rows, m.n_cols);
  result.zeros();

  for (uint32_t i = 0 ; i < m.n_rows ; ++i) {
    double sum = 0;
    for (uint32_t j = 0 ; j < m.n_cols ; ++j) {
      sum += m(i, j);
    }
    for (uint32_t j = 0 ; j < m.n_cols ; ++j) {
      result(i, j) = sum;
    }
  }
  return result;
}

mat cost(mat y, mat h) {
  //const mat exponential = nn::funcop(h, exp);
  //const mat sum = colsum(exponential);
  //const mat p = exponential % (1 / sum);
  //const mat J = - (y % nn::funcop(p, log));
  //mat J = -(y % nn::funcop(h, log) + (1-y) % nn::funcop(1-h, log));
  const mat diff = y - h;
  const mat J = (diff % diff) / 2.0;
  //mat J = -(y % h.transform(log) + (1-y) % nn::funcop(1-h, log)) / y.n_rows;
  //mat J = -(y % h);
  return J;
}

mat costd(mat y, mat a, mat,mat) {
  //const mat exponential = nn::funcop(a, exp);
  //const mat sum = colsum(exponential);
  //const mat p = exponential % (1.0 / sum);
  //const mat delta = p - y;
  //mat delta = (a - y);
  const mat delta = (a - y);
  //mat delta = (a - y) / y.n_rows;
  //mat grad = -(y % nn::funcop(z, sigmoidgrad));
  return delta;
}

void load(mat &x, mat &y) {
  std::ifstream input("/home/joseph/C/data/house.dat", std::ios::in);
  std::string line;

  if (input.is_open()) {
    x = mat(datasize, n);
    y = mat(datasize, o);
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
        if (j == 0) xi /= (7322564.0 / 10.0);

        x(i, j) = xi;

        // read ,
        input >> c;
      }

      double val = 0;
      input >> val;
      y(i, 0) = val / scale;
    }
  }

  // random the input data
  cout << "shuffle data..." << endl;
  for (uint32_t i = 0 ; i < x.n_rows ; ++i) {
    int index1 = rand() % x.n_rows;
    int index2 = rand() % x.n_rows;
    mat xrow1 = x.row(index1);
    mat xrow2 = x.row(index2);
    mat yrow1 = y.row(index1);
    mat yrow2 = y.row(index2);
    for (uint32_t j = 0 ; j < x.n_cols ; ++j) {
      x(index1, j) = xrow2(0, j);
      x(index2, j) = xrow1(0, j);
    }

    for (uint32_t j = 0 ; j < y.n_cols ; ++j) {
      y(index1, j) = yrow2(0, j);
      y(index2, j) = yrow1(0, j);
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

mglData prediction;
mglData answer;
int sample(mglGraph* graph) {
  graph->AddLegend("prediction", "r");
  graph->AddLegend("answer", "b");
  graph->SetRanges(0, prediction.nx + 10, 0, 500001);
  graph->Axis("y");
  graph->Label('y', "prices");
  graph->Plot(answer, "-b2");
  graph->Plot(prediction, "-r2");
  return 0;
}

int main() {
  double lrate = 6e-6;
  //const double lratedecay = 0.99;
  const double lambda = 0;
  //const int batchsize = 1000;

  srand(time(NULL));

  InputLayer input(n);
  vector<Layer> hidden = {
    //Layer(n, 32, lrate, lambda, atan, [] (double x) {return 1.0/(1.0+x*x);}),
    //Layer(32, 8, lrate, lambda, atan, [] (double x) {return 1.0/(1.0+x*x);}),
    Layer(n, 64, lrate, lambda, sigmoid, sigmoidgrad),
  };
  OutputLayer output(64, o, lrate, 0, identity, identitygrad, cost, costd);
  NeuralNet nnet(input, output, hidden);

  mat x, y;
  load(x, y);
  //cout << x.n_rows << endl;
  //cout << y.n_rows << endl;

  //nnet.feeddata(x.row(1), y.row(1), true);
  for (int i = 0 ; i < datasize * 60 ; ++i) {
    //const int start = i % (datasize-batchsize);
    //const int end = start + batchsize;
    //nnet.feeddata(x.rows(start, end), y.rows(start, end), false);
    nnet.feeddata(x.row(i%datasize), y.row(i%datasize), false);
    //nnet.feeddata(x, y, false);
    cout << "\riteration: " << i+1 << " cost: " << nnet.computecost();
    if (i % datasize == 0) {
      cout << endl << "iteration: " << i+1 << " cost: " << nnet.computecost();
      //cout << endl << nnet.getresult() << endl;
      //cout << y.row(i% datasize) << endl;
    }
    //if (i % 500 == 0) {
      //lrate *= lratedecay;
      //nnet.setlrate(lrate);
    //}
  }
  cout << endl;
  mat result = nnet.predict(x);
  //for (uint32_t i = 0 ; i < y.n_rows ; ++i) {
    //cout << "predict: " << result(i, 1) * scale <<
        //" | answer: " << y(i, 0) * scale << endl;
  //}

  const int sampling = 10;
  prediction = mglData(result.n_rows/sampling);
  answer = mglData(y.n_rows/sampling);
  for (uint32_t i = 0 ; i < y.n_rows/sampling ; ++i) {
    prediction.a[i] = result(i/sampling, 1) * scale;
    answer.a[i] = y(i/sampling, 0) * scale;
  }
  mglQT display(sample, "house");

  return display.Run();
}
