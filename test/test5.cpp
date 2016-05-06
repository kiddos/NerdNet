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
  std::ifstream input("/home/joseph/C/data/iris.dat", std::ios::in);
  std::string line;
  const int datasize = 150;
  const double trainpercent = 0.8;

  if (input.is_open()) {
    mat x = mat(datasize, 4);
    mat y = mat(datasize, 3);
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

    shuffledata(x, y);
    const int trainsize = static_cast<int>(datasize * trainpercent);
    const int testsize = datasize - trainsize;

    trainx = mat(trainsize, x.n_cols);
    trainy = mat(trainsize, y.n_cols);
    testx = mat(testsize, x.n_cols);
    testy = mat(testsize, y.n_cols);

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
  double lrate = 6e-4;
  //const double lratedecay = 0.90;
  const double lambda = 1e-8;

  srand(time(NULL));

  InputLayer input(4);
  vector<Layer> hidden = {
    Layer(4, 6, lrate, lambda, nn::sigmoid),
    Layer(6, 6, lrate, lambda, nn::sigmoid),
    Layer(6, 6, lrate, lambda, nn::sigmoid),
  };
  //OutputLayer output(4, 3, lrate, 0, sigmoid, sigmoidgrad, cost, costd);
  //nn::CrossEntropyOutput output(6, 3, lrate, lambda);
  //nn::SoftmaxOutput output(6, 3, lrate, lambda);
  //nn::QuadraticOutput output(6, 3, lrate, lambda);
  nn::KullbackLeiblerOutput output(6, 3, lrate, lambda);
  NeuralNet nnet(input, output, hidden);

  mat trainx, trainy, testx, testy;
  load(trainx, trainy, testx, testy);
  cout << trainx << endl;
  cout << trainy << endl;

  //nnet.feeddata(x, y, true);
  for (uint32_t i = 0 ; i < trainx.n_rows * 1000 ; ++i) {
    nnet.feeddata(trainx, trainy, false);
    //nnet.feeddata(x.row(i%x.n_rows), y.row(i%x.n_rows), false);
    const double newcost = nnet.computecost();
    cout << "\riteration: " << i+1 << " cost: " << newcost;

    if (i % (trainx.n_rows * 10) == 0) {
      cout << endl << "iteration: " << i+1 << " cost: " << newcost;
    }

    //if (newcost < 150 && i % (x.n_rows * 10) == 0) {
      //if (newcost < 50) {
        //lrate *= 0.96;
      //} else {
        //lrate *= lratedecay;
      //}
      //nnet.setlrate(lrate);
    //}
  }
  cout << endl;
  mat result = nnet.predict(trainx);
  cout << "train accuracy: " << accuracy(trainy, result) * 100 << endl;
  result = nnet.predict(testx);
  cout << "test accuracy: " << accuracy(testy, result) * 100 << endl;

  return 0;
}
