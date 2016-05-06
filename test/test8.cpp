#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string>

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
const double trainpercent = 0.8;
//const int datasize = 300;
const int n = 5;

void shuffledata(mat &x, mat &y) {
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

void load(mat &trainx, mat &trainy, mat &testx, mat &testy) {
  std::ifstream input("/home/joseph/C/data/phoneme.dat", std::ios::in);
  std::string line;

  if (input.is_open()) {
    mat x = mat(datasize, n);
    mat y = mat(datasize, 2);
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

      int val = 0;
      input >> val;
      y(i, val) = 1;
    }

    const int trainsize = static_cast<int>(datasize * trainpercent);
    const int testsize = datasize - trainsize;

    trainx = mat(trainsize, x.n_cols);
    trainy = mat(trainsize, y.n_cols);
    testx = mat(testsize, x.n_cols);
    testy = mat(testsize, y.n_cols);

    shuffledata(x, y);
    for (int i = 0 ; i < trainsize ; ++i) {
      for (uint32_t j = 0 ; j < trainx.n_cols ; ++j) {
        trainx(i, j) = x(i, j);
      }
      for (uint32_t j = 0 ; j < trainy.n_cols ; ++j) {
        trainy(i, j) = y(i, j);
      }
    }

    for (int i = 0 ; i < testsize ; ++i) {
      for (uint32_t j = 0 ; j < trainx.n_cols ; ++j) {
        testx(i, j) = x(i+trainsize, j);
      }
      for (uint32_t j = 0 ; j < trainy.n_cols ; ++j) {
        testy(i, j) = y(i+trainsize, j);
      }
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
  const double lrate0 = 1e-2;
  const double lambda = 0;
  const double decayfactor = datasize;
  double lrate = lrate0;
  const int hiddennodes = 4096;

  srand(time(NULL));

  InputLayer input(5);
  vector<Layer> hidden = {
    Layer(5, hiddennodes, lrate, lambda, nn::arctan),
  };
  nn::CrossEntropyOutput output(hiddennodes, 2, lrate, lambda);
  NeuralNet nnet(input, output, hidden);

  mat trainx, trainy, testx, testy;
  load(trainx, trainy, testx, testy);
  //cout << x.n_rows << endl;
  //cout << y.n_rows << endl;

  //const int batchsize = x.n_rows / 20;
  //const int batchsize = 300;
  //nnet.feeddata(x.row(0), y.row(0), true);
  for (uint32_t i = 0 ; i < trainx.n_rows * 25 ; ++i) {
    //const int start = i % (x.n_rows-batchsize);
    //const int end = start + batchsize;
    //nnet.feeddata(x.rows(start, end), y.rows(start, end), false);
    //nnet.feeddata(x, y, false);
    nnet.feeddata(trainx.row(i % trainx.n_rows), trainy.row(i % trainy.n_rows), false);
    cout << "\riteration: " << i+1 << " cost: " << nnet.computecost()
          << "          ";
    if (i % (trainx.n_rows * 1) == 0) {
      cout << endl << "iteration: " << i+1 << " cost: " << nnet.computecost()
          << "          ";
    }
    if (i % (trainx.n_rows) == 0) {
      lrate = lrate0 * exp(-1.0 * i / decayfactor);
      nnet.setlrate(lrate);
    }
  }
  cout << endl;
  mat result = nnet.predict(trainx);
  //cout << nnet.getresult() << endl;
  cout << "train accuracy: " << accuracy(trainy, result) * 100 << endl;
  result = nnet.predict(testx);
  cout << "accuracy: " << accuracy(testy, result) * 100 << endl;

  return 0;
}
