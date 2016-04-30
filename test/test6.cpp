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
  const mat diff = y - h;
  const mat J = (diff % diff) / 2.0;
  //mat J = -(y % h.transform(log) + (1-y) % nn::funcop(1-h, log));
  //mat J = -(y % h.transform(log) + (1-y) % nn::funcop(1-h, log)) / y.n_rows;
  //mat J = -(y % h);
  return J;
}

mat costd(mat y, mat a, mat) {
  const mat grad = (a - y);
  //mat grad = (a - y);
  //mat grad = (a - y) / y.n_rows;
  //mat grad = -(y % nn::funcop(z, sigmoidgrad));
  return grad;
}

void load(mat &x, mat &y) {
  std::ifstream input("/home/joseph/C/data/diabetes.dat", std::ios::in);
  std::string line;
  const int datasize = 43;

  if (input.is_open()) {
    x = mat(datasize, 2);
    y = mat(datasize, 1);
    y.zeros();

    for (; std::getline(input, line) ;) {
      if (strcmp(line.substr(0, 5).c_str(), "@data") == 0) {
        cout << "reading data..." << endl;
        break;
      }
    }

    for (uint32_t i = 0 ; i < datasize ; ++i) {
      for (int j = 0 ; j < 2 ; ++j) {
        double xi = 0;
        char c;
        input >> xi;
        x(i, j) = xi;

        input >> c;
      }

      double value = 0;
      input >> value;
      y(i, 0) = value;
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
  graph->SetRanges(0, prediction.nx + 10, 0, 7);
  graph->Axis("y");
  graph->Label('y', "C peptide");
  graph->Plot(prediction, "-r5");
  graph->Plot(answer, "-b5");
  return 0;
}

int main() {
  const double lrate = 1e-3;
  const double lambda = 0;

  srand(time(NULL));

  InputLayer input(2);
  vector<Layer> hidden = {
    Layer(2, 100, lrate, lambda, sigmoid, sigmoidgrad),
  };
  OutputLayer output(100, 1, lrate, lambda, identity, identitygrad, cost, costd);
  NeuralNet nnet(input, output, hidden);

  mat x, y;
  load(x, y);
  //cout << x << endl;
  //cout << y << endl;

  //nnet.feeddata(x, y, true);
  for (int i = 0 ; i < 66000 ; ++i) {
    nnet.feeddata(x, y, false);
    cout << "\riteration: " << i+1 << " cost: " << nnet.computecost();
  }
  cout << endl;
  mat result = nnet.predict(x);
  //for (uint32_t i = 0 ; i < y.n_rows ; ++i) {
    //cout << "predict: " << result(i, 1) << " | answer: " << y(i, 0) << endl;
  //}

  prediction = mglData(result.n_rows);
  answer = mglData(y.n_rows);
  for (uint32_t i = 0 ; i < y.n_rows ; ++i) {
    prediction.a[i] = result(i, 1);
    answer.a[i] = y(i, 0);
  }
  mglQT display(sample, "diabetes");

  return display.Run();
}
