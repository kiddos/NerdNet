#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/shape.hpp>

#include <mgl2/mgl.h>

#include "nnet.h"

using std::ofstream;
using std::vector;
using std::string;
using std::cout;
using std::cin;
using std::endl;

using nn::mat;
using nn::Layer;
using nn::InputLayer;
using nn::OutputLayer;
using nn::NeuralNet;
using nn::Trainer;
using nn::ActFunc;

const double trainpercent = 0.8;

typedef struct Accuracy {
  double train, test;
} Accuracy;

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
void load(mat& trainx, mat& trainy, mat& testx, mat& testy) {
  int n = 0, m = 0, o = 0;
  // read number of feature,
  // number of output,
  // number of training samples
  // in this order
  cin >> n;
  cin >> o;
  cin >> m;
  mat x = mat(m, n);
  mat y = mat(m, o);
  y.zeros();

  double val = 0;
  for (int i = 0 ; i < m ; ++i) {
    for (int j = 0 ; j < n ; ++j) {
      cin >> val;
      x(i, j) = val;
    }

    cin >> val;
    y(i, static_cast<int>(val)) = 1;
  }
  shuffledata(x, y);
  const int trainsize = static_cast<int>(x.n_rows * trainpercent);

  trainx = x.submat(0, 0, trainsize-1, x.n_cols-1);
  trainy = y.submat(0, 0, trainsize-1, y.n_cols-1);

  testx = x.submat(trainsize, 0, x.n_rows-1, x.n_cols-1);
  testy = y.submat(trainsize, 0, y.n_rows-1, y.n_cols-1);
}
void buildmodel(NeuralNet& nnet, int n, int o, ofstream& outputcsv) {
  InputLayer input(n);
  OutputLayer output;
  vector<Layer> hidden;
  int nhidden = 0;

  int lastnnode = n;
  cin >> nhidden;
  outputcsv << "number of hidden," << nhidden << endl;
  for (int i = 0 ; i < nhidden ; ++i) {
    double lrate = 0, lambda = 0;
    int nnodes = 0;
    string actfunc;
    cin >> lrate;
    cin >> lambda;
    cin >> nnodes;
    cin >> actfunc;

    ActFunc act;
    if (strcmp(actfunc.c_str(), "arctan") == 0) {
      act = nn::arctan;
    } else if (strcmp(actfunc.c_str(), "sigmoid") == 0) {
      act = nn::sigmoid;
    } else if (strcmp(actfunc.c_str(), "tanh") == 0) {
      act = nn::tanhyperbolic;
    } else if (strcmp(actfunc.c_str(), "hardtanh") == 0) {
      act = nn::hardtanh;
    } else if (strcmp(actfunc.c_str(), "identity") == 0) {
      act = nn::identity;
    } else if (strcmp(actfunc.c_str(), "relu") == 0) {
      act = nn::relu;
    } else if (strcmp(actfunc.c_str(), "absolute") == 0) {
      act = nn::absolute;
    } else if (strcmp(actfunc.c_str(), "relucos") == 0) {
      act = nn::relucos;
    } else if (strcmp(actfunc.c_str(), "relusin") == 0) {
      act = nn::relusin;
    } else if (strcmp(actfunc.c_str(), "smoothrelu") == 0) {
      act = nn::smoothrelu;
    } else {
      cout << "ERROR!! Unknown activation function" << endl;
      act = nn::arctan;
    }

    cout << "hidden " << i << " size: " << lastnnode << ", " << nnodes
        << " activation: " << actfunc << endl;
    outputcsv << "hidden " << i << "," << lastnnode << ", " << nnodes
        << "," << actfunc << endl;
    hidden.push_back(Layer(lastnnode, nnodes, lrate, lambda, act));
    lastnnode = nnodes;
  }

  double lrate = 0, lambda = 0;
  string outputtype;
  cin >> outputtype;
  cin >> lrate;
  cin >> lambda;
  cout << "output w size: " << lastnnode << ", " << o
      << " type: " << outputtype << endl;
  if (strcmp(outputtype.c_str(), "softmax") == 0) {
    output = nn::SoftmaxOutput(lastnnode, o, lrate, lambda);
  } else if (strcmp(outputtype.c_str(), "quadratic") == 0) {
    output = nn::QuadraticOutput(lastnnode, o, lrate, lambda);
  } else if (strcmp(outputtype.c_str(), "crossentropy") == 0) {
    output = nn::CrossEntropyOutput(lastnnode, o, lrate, lambda);
  } else if (strcmp(outputtype.c_str(), "kullbackleibler") == 0) {
    output = nn::KullbackLeiblerOutput(lastnnode, o, lrate, lambda);
  }

  nnet = NeuralNet(input, output, hidden);
}
void setuptrainer(NeuralNet& nnet, Trainer& trainer,
                  unsigned long& maxiters, ofstream& outputcsv) {
  string name;
  unsigned long iters = 0;
  double r0 = 0, k = 0;
  int step = 0;
  cin >> name;
  cin >> iters;
  cin >> r0;
  cin >> k;
  cin >> step;

  if (strcmp(name.c_str(), "gd")) {
    trainer = Trainer(nnet, r0, k, step);
  } else if (strcmp(name.c_str(), "sgd")) {
    trainer = nn::SGDTrainer(nnet, r0, k, step);
  } else if (strcmp(name.c_str(), "batch")) {
    int batchsize = 0;
    cin >> batchsize;
    trainer = nn::BatchTrainer(nnet, batchsize, r0, k, step);
  } else if (strcmp(name.c_str(), "momentum")) {
    double momentum = 0;
    cin >> momentum;
    trainer = nn::MomentumTrainer(nnet, momentum, r0, k, step);
  }

  maxiters = iters;
  cout << "trainer setup: " << name << " | maxiters: " << maxiters << endl;
  outputcsv << name << "," << maxiters << "," << r0 << "," << k
      << "," << step << endl;
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
void train(Trainer& trainer, NeuralNet& nnet,
           mat& x, mat& y, unsigned long maxiters, mglData& data) {
  vector<double> costdata;
  time_t start = time(nullptr);
  time_t pass = time(nullptr) - start;
  for (uint32_t i = 0 ; i < maxiters ; ++i) {
    if (i % 10 == 0 || pass > 1) {
      const double cost = trainer.feeddata(x, y, true);
      costdata.push_back(cost);

      cout << "\riteration: " << i << " | cost: " << cost;
      mat pred = nnet.predict(x);
      cout << " | training accuracy: " << accuracy(y, pred);

      start = time(nullptr);
    } else {
      trainer.feeddata(x, y, false);
    }
    pass = time(nullptr) - start;
  }
  cout << endl << "Training Complete!!" << endl;

  data = mglData(costdata.size());
  for (uint32_t i = 0 ; i < costdata.size() ; ++i) {
    data.a[i] = costdata[i];
  }
}
Accuracy evaluate(NeuralNet& nnet, mat& trainx, mat& trainy,
                  mat& testx, mat& testy) {
  Accuracy accu;
  mat pred = nnet.predict(trainx);
  accu.train = accuracy(trainy, pred);

  pred = nnet.predict(testx);
  accu.test = accuracy(testy, pred);
  return accu;
}
void draw(mglGraph* graph, const mglData& data,
          unsigned long maxiters, string outputfile) {
  graph->AddLegend("cost", "b");
  graph->SetRanges(0, maxiters, 0, data.a[0]*2);
  graph->Axis("y");
  graph->Axis("x");
  graph->Label('y', "cost");
  graph->Label('x', "epoch");
  graph->Plot(data, "-b5");

  graph->WritePNG(outputfile.c_str());
}
void savedata() {
}

int main(void) {
  NeuralNet nnet, raw;
  Trainer trainer;
  unsigned long maxiters = 0;
  int ntrail = 0;
  string name;
  cin >> name;
  ofstream outputcsv(name+".csv", std::ios::out);

  mat trainx, trainy, testx, testy;
  load(trainx, trainy, testx, testy);

  srand(time(NULL));

  //cout << trainx << endl;
  //cout << trainy << endl;
  cout << "train x size: " << trainx.n_rows << ", " << trainx.n_cols << endl;
  cout << "train y size: " << trainy.n_rows << ", " << trainy.n_cols << endl;
  cout << "test x size: " << testx.n_rows << ", " << testx.n_cols << endl;
  cout << "test y size: " << testy.n_rows << ", " << testy.n_cols << endl;

  // setup
  buildmodel(raw, trainx.n_cols, trainy.n_cols, outputcsv);
  nnet = raw;
  setuptrainer(nnet, trainer, maxiters, outputcsv);
  mglData data;
  string trailname;

  cin >> ntrail;
  for (int i = 0 ; i < ntrail ; ++i) {
    cout << "trail " << (i+1) << endl;
    train(trainer, nnet, trainx, trainy, maxiters, data);

    Accuracy accu = evaluate(nnet, trainx, trainy, testx, testy);
    cout << "Train accuracy: " << accu.train << endl;
    cout << "Test accuracy: " << accu.test << endl;

    outputcsv << "Trail" << i+1 << endl;
    outputcsv << "Train accuracy," << accu.train << endl;
    outputcsv << "Test accuracy," << accu.test << endl;

    // save the graph
    mglGraph graph;
    trailname = name + "-trail-" + std::to_string(i) + ".png";
    draw(&graph, data, maxiters, trailname);

    nnet = raw;
    for (uint32_t j = 0 ; j < nnet.getnumhidden() * 2 ; ++j) {
      nnet.randomize();
    }
  }
  cout << "All trail completed!!" << endl;

  outputcsv.close();
  return 0;
}
