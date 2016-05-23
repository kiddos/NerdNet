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
using nn::mat;
using nn::Layer;
using nn::InputLayer;
using nn::OutputLayer;
using nn::NeuralNet;
using nn::Trainer;
using nn::BatchTrainer;

const int datasize = 660;
const double trainingpercent = 0.66;
const int n = 66;
const int o = 2;

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
  std::ifstream input("/home/joseph/C/data/game.input", std::ios::in);
  std::string line;

  const int trainingsize = static_cast<int>(trainingpercent * datasize);
  const int testingsize = datasize - trainingsize;
  if (input.is_open()) {
    trainx = mat(trainingsize, n);
    trainy = mat(trainingsize, o);
    testx = mat(testingsize, n);
    testy = mat(testingsize, o);
    trainy.zeros();
    testy.zeros();

    cout << "loading data for training..." << endl;
    for (uint32_t i = 0 ; i < trainingsize ; ++i) {
      for (int j = 0 ; j < n ; ++j) {
        double xi = 0;
        input >> xi;
        trainx(i, j) = xi;
      }

      double val = 0;
      input >> val;
      trainy(i, val) = 1;
    }

    cout << "loading data for testing..." << endl;
    for (uint32_t i = 0 ; i < testingsize ; ++i) {
      for (int j = 0 ; j < n ; ++j) {
        double xi = 0;
        input >> xi;
        testx(i, j) = xi;
      }

      double val = 0;
      input >> val;
      testy(i, val) = 1;
    }
  }

  // shuffle
  cout << "shuffle data..." << endl;
  //shuffledata(trainx, trainy);
  //shuffledata(testx, testy);
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
  //const double lrate0 = 5e-2;
  const double lrate0 = 1e-3;
  const double lambda = 1e-6;
  const int batchsize = 60;

  srand(time(NULL));


  InputLayer input(n);
  vector<Layer> hidden = {
    Layer(n, 100, lrate0, lambda, nn::sigmoid),
    Layer(100, 50, lrate0, lambda, nn::sigmoid),
    Layer(50, 25, lrate0, lambda, nn::sigmoid),
    //Layer(n, n, lrate, lambda, nn::sigmoid),
    //Layer(n, n, lrate, lambda, nn::sigmoid),
    //Layer(n, n, lrate, lambda, nn::sigmoid),
    //Layer(n, n, lrate, lambda, nn::sigmoid),
  };
  nn::SoftmaxOutput output(25, o, lrate0, lambda);
  NeuralNet nnet(input, output, hidden);
  BatchTrainer trainer(nnet, batchsize, lrate0, 1e-5, batchsize*10);

  mat trainx, trainy, testx, testy;
  load(trainx, trainy, testx, testy);
  cout << trainx.n_rows << endl;
  cout << trainy.n_rows << endl;

  trainer.gradcheck(trainx.row(1), trainy.row(1));
  for (int i = 0 ; i < datasize * 30 ; ++i) {
    const double newcost = trainer.feeddata(trainx, trainy, true);
    cout << "\riteration: " << i+1 << " cost: " << newcost;

    if (i % 10 == 0) {
      mat result = nnet.predict(trainx);
      cout << endl << "accuracy: " << accuracy(trainy, result) * 100 << endl;
    }
  }

  cout << endl;
  mat result = nnet.predict(testx);
  cout << "testing accuracy: " << accuracy(testy, result) * 100 << endl;
  result = nnet.predict(trainx);
  //cout << nnet.getresult() << endl;
  cout << "training accuracy: " << accuracy(trainy, result) * 100 << endl;

  return 0;
}
