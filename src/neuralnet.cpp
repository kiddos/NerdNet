#include "neuralnet.h"

using std::vector;
using std::shared_ptr;
using std::string;
using std::ifstream;
using std::ofstream;

namespace nn {

NeuralNet::NeuralNet() : eps(1e-4) {}

NeuralNet::NeuralNet(const NeuralNet& nnet)
    : eps(nnet.eps), result(nnet.result),
      cost(nnet.cost), costd(nnet.costd),
      input(nnet.input), hidden(nnet.hidden), output(nnet.output) {}

NeuralNet::NeuralNet(const InputLayer input, const OutputLayer output,
                     vector<shared_ptr<Layer>> hidden)
    : eps(1e-4), cost(output.getcost()), costd(output.getcostd()),
      input(input), hidden(hidden), output(output) {}

NeuralNet& NeuralNet::operator= (const NeuralNet& nnet) {
  result = nnet.result;
  cost = nnet.cost;
  costd = nnet.costd;
  input = nnet.input;
  hidden = nnet.hidden;
  output = nnet.output;
  return *this;
}

mat NeuralNet::forwardprop(const mat& x) {
  mat current = input.forwardprop(x);
  for (uint32_t i = 0 ; i < hidden.size() ; ++i) {
    const mat n = hidden[i]->forwardprop(current);
    current = n;
  }
  result = output.forwardprop(current);
  return result;
}

void NeuralNet::backprop(const mat& y) {
  mat currentdelta = output.backprop(y);
  for (int i = hidden.size()-1 ; i >= 0 ; --i) {
    mat p = hidden[i]->backprop(currentdelta);
    currentdelta = p;
  }
}

void NeuralNet::update() {
  output.update();
  for (uint32_t i = 0 ; i < hidden.size() ; ++i) {
    hidden[i]->update();
  }
}

void NeuralNet::update(const mat& ograd, const vector<mat>& hgrad) {
  output.update(ograd);
  for (uint32_t i = 0 ; i < hidden.size() ; ++i) {
    hidden[i]->update(hgrad[i]);
  }
}

mat NeuralNet::predict(const mat& sample) {
  // prediction is simply forwardprop
  result = forwardprop(sample);
  mat argmax = output.argmax();
  return argmax;
}

double NeuralNet::computecost(const mat& pred, const mat& y) {
  mat J = cost(y, pred);
  return sumall(J);
}

void NeuralNet::setlrate(const double lrate) {
  input.setlrate(lrate);
  for (uint32_t i = 0 ; i < hidden.size() ; ++i)
    hidden[i]->setlrate(lrate);
  output.setlrate(lrate);
}

void NeuralNet::randomize(const double stddev) {
  int index = rand() % (hidden.size()+1);
  randomize(index, stddev);
}

void NeuralNet::randomize(const uint32_t index, const double stddev) {
  if (index < hidden.size()) {
    const mat w = hidden[index]->getw();
    hidden[index]->randominit(stddev);
  } else {
    const mat w = output.getw();
    output.randominit(stddev);
  }
}

void NeuralNet::save(const string path) {
  ofstream out(path, std::ios::out);
  if (out.is_open()) {
    // save hidden layer parameter first
    for (uint32_t i = 0 ; i < hidden.size() ; ++i) {
      const mat w = hidden[i]->getw();
      for (uint32_t j = 0 ; j < w.n_rows ; ++j) {
        for (uint32_t k = 0 ; k < w.n_cols ; ++k) {
          out << w(j, k) << " ";
        }
      }
      out << "\n";
    }
    // save output layer
    const mat w = output.getw();
    for (uint32_t i = 0 ; i < w.n_rows ; ++i) {
      for (uint32_t j = 0 ; j < w.n_cols ; ++j) {
        out << w(i, j) << " ";
      }
    }
    out << "\n";
    out.close();
  }
}

void NeuralNet::load(const string path) {
  ifstream in(path, std::ios::in);
  if (in.is_open()) {
    // read hidden layer first
    for (uint32_t i = 0 ; i < hidden.size() ; ++i) {
      mat w = hidden[i]->getw();
      for (uint32_t j = 0 ; j < w.n_rows ; ++j) {
        for (uint32_t k = 0 ; k < w.n_cols ; ++k) {
          double val = 0;
          in >> val;
          w(j, k) = val;
        }
      }
      hidden[i]->setw(w);
    }
    // load output layer
    mat w = output.getw();
    for (uint32_t i = 0 ; i < w.n_rows ; ++i) {
      for (uint32_t j = 0 ; j < w.n_cols ; ++j) {
        double val = 0;
        in >> val;
        w(i, j) = val;
      }
    }
    output.setw(w);

    in.close();
  }
}

}
