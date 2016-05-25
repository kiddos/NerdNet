#include "neuralnet.h"

using std::vector;
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
                     vector<Layer> hidden)
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
    const mat n = hidden[i].forwardprop(current);
    current = n;
  }
  result = output.forwardprop(current);
  return result;
}

void NeuralNet::backprop(const mat& y) {
  mat currentdelta = output.backprop(y);
  for (int i = hidden.size()-1 ; i >= 0 ; --i) {
    mat p = hidden[i].backprop(currentdelta);
    currentdelta = p;
  }
}

void NeuralNet::update() {
  output.update();
  for (uint32_t i = 0 ; i < hidden.size() ; ++i) {
    hidden[i].update();
  }
}

void NeuralNet::update(const mat& ograd, const vector<mat>& hgrad) {
  output.update(ograd);
  for (uint32_t i = 0 ; i < hidden.size() ; ++i) {
    hidden[i].update(hgrad[i]);
  }
}

mat NeuralNet::predict(const mat& sample) {
  // prediction is simply forwardprop
  result = forwardprop(sample);
  mat argmax = output.argmax();
  return argmax;
}

bool NeuralNet::gradcheck(const mat& x, const mat& y) {
  forwardprop(x);
  backprop(y);

  // back prop result from output to input
#ifdef DEBUG
  cout << "gradient checking ......";
#endif

  bool success = true;
  mat w = output.getw();
  mat ngrad = computengrad(x, y, w.n_rows, w.n_cols, hidden.size());
  mat grad = output.getgrad();

  if (!issame(grad, ngrad)) {
    if (success) {
#ifdef DEBUG
      cout << "failed" << endl;
#endif

      success = false;
    }
#ifdef DEBUG
    cout << "output layer: " << endl;
    cout << "backprop grad:" << endl << grad << endl;
    cout << "numeric grad:" << endl << ngrad << endl;
#endif
  }

  for (int i = hidden.size()-1 ; i >= 0 ; --i) {
    w = hidden[i].getw();
    ngrad = computengrad(x, y, w.n_rows, w.n_cols, i);
    grad = hidden[i].getgrad();

    if (!issame(grad, ngrad)) {
      if (success) {
#ifdef DEBUG
        cout << "failed" << endl;
#endif

        success = false;
      }
#ifdef DEBUG
      cout << "hidden " << i << ":" << endl;
      cout << "backprop grad:" << endl << grad << endl;
      cout << "numeric grad:" << endl << ngrad << endl;
#endif

    }
  }

  if (success) {
#ifdef DEBUG
    cout << " passed." << endl;
#endif
  }

  return success;
}

double NeuralNet::computecost(const mat& pred, const mat& y) {
  mat J = cost(y, pred);
  return sumall(J);
}

void NeuralNet::setlrate(const double lrate) {
  input.setlrate(lrate);
  for (uint32_t i = 0 ; i < hidden.size() ; ++i)
    hidden[i].setlrate(lrate);
  output.setlrate(lrate);
}

bool NeuralNet::issame(const mat& m1, const mat& m2) {
  const double scale = 1.0 / eps;
  for (uint32_t i = 0 ; i < m1.n_rows ; ++i) {
    for (uint32_t j = 0 ; j < m1.n_cols ; ++j) {
      const double val1 = m1(i, j) * scale;
      const double val2 = m2(i, j) * scale;
      if (fabs(val1 - val2) > 10.0) {
#ifdef DEBUG
        cout << " diff at index(" << i << "," << j << "): diff: " <<
            fabs(val1-val2) <<
            "val1: " << val1 << ", val2:" << val2 << endl;
#endif
        return false;
      }
    }
  }
  return true;
}

double NeuralNet::computecost(const mat& x, const mat&y,
                              const mat& perturb, uint32_t idx) {
  if (idx > hidden.size()) return DBL_MAX;

  // forward propagation
  mat current = input.forwardprop(x);
  for (uint32_t i = 0 ; i < hidden.size() ; ++i) {
    if (i == idx) {
      mat tempw = hidden[i].getw();

      hidden[i].setw(tempw + perturb);
      mat n = hidden[i].forwardprop(current);
      current = n;
      hidden[i].setw(tempw);
    } else {
      mat n = hidden[i].forwardprop(current);
      current = n;
    }
  }
  mat out;
  if (idx == hidden.size()) {
    mat tempw = output.getw();
    output.setw(tempw + perturb);
    out = output.forwardprop(current);
    output.setw(tempw);
  } else {
    out = output.forwardprop(current);
  }

  mat J = cost(y, out);
  double val = 0;
  for (uint32_t i = 0 ; i < J.n_rows ; ++i) {
    for (uint32_t j = 0 ; j < J.n_cols ; ++j) {
      val += J(i, j);
    }
  }

  // regularization
  for (uint32_t i = 0 ; i < hidden.size() ; ++i) {
    mat tempw = hidden[i].getw();
    if (i == idx) {
      tempw = tempw + perturb;
    }

    const mat regterm = (hidden[i].getlambda() / 2.0) * (tempw % tempw);
    for (uint32_t j = 0 ; j < regterm.n_rows ; ++j) {
      for (uint32_t k = 0 ; k < regterm.n_cols ; ++k) {
        val += regterm(j, k);
      }
    }
  }

  mat tempw = output.getw();
  if (idx == hidden.size()) {
    tempw = tempw + perturb;
  }
  const mat regterm = (output.getlambda() / 2.0) * (tempw % tempw);
  for (uint32_t j = 0 ; j < regterm.n_rows ; ++j) {
    for (uint32_t k = 0 ; k < regterm.n_cols ; ++k) {
      val += regterm(j, k);
    }
  }
  return val;
}

mat NeuralNet::computengrad(const mat&x, const mat&y,
                            int nrows, int ncols, int idx) {
  mat wgrad(nrows, ncols);
  mat perturb(nrows, ncols);
  wgrad.zeros();
  perturb.zeros();

  for (int i = 0 ; i < nrows ; ++i) {
    for (int j = 0 ; j < ncols ; ++j) {
      perturb(i, j) = eps;
      const double loss1 = computecost(x, y, perturb, idx);
      const double loss2 = computecost(x, y, -perturb, idx);
      wgrad(i, j) = (loss1 - loss2) / (2 * eps);

      perturb(i, j) = 0;
    }
  }

  return wgrad;
}

void NeuralNet::randomize() {
  int index = rand() % (hidden.size()+1);
  randomize(index);
}

void NeuralNet::randomize(uint32_t index) {
  if (index < hidden.size()) {
    const mat w = hidden[index].getw();
    hidden[index] = Layer(hidden[index]);
  } else {
    const mat w = output.getw();
    output.randominit(sqrt(w.n_rows));
  }
}

void NeuralNet::save(const string path) {
  ofstream out(path, std::ios::out);
  if (out.is_open()) {
    // save hidden layer parameter first
    for (uint32_t i = 0 ; i < hidden.size() ; ++i) {
      const mat w = hidden[i].getw();
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
      mat w = hidden[i].getw();
      for (uint32_t j = 0 ; j < w.n_rows ; ++j) {
        for (uint32_t k = 0 ; k < w.n_cols ; ++k) {
          double val = 0;
          in >> val;
          w(j, k) = val;
        }
      }
      hidden[i].setw(w);
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
