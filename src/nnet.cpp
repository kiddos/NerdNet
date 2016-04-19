#include "nnet.h"

#ifdef DEBUG
#include <iostream>

using std::cout;
using std::endl;
#endif

using std::vector;

namespace nn {

NeuralNet::NeuralNet(const InputLayer input, const OutputLayer output,
                     vector<Layer> hidden) :
                     eps(1e-5), cost(output.getcost()), costd(output.getcostd()),
                     input(input), hidden(hidden), output(output) {}

void NeuralNet::feeddata(const mat x, const mat y, const bool check) {
  this->x = x;
  this->y = y;

  // forward propagation
  mat current = input.forwardprop(x);
  for (uint32_t i = 0 ; i < hidden.size() ; ++i) {
    mat n = hidden[i].forwardprop(current);
    current = n;
  }
  result = output.forwardprop(current);

  // backpropagation
  mat currentdelta = output.backprop(y);
  for (int i = hidden.size()-1 ; i >= 0 ; --i) {
    mat p = hidden[i].backprop(currentdelta);
    currentdelta = p;
  }

  if (check) gradcheck();

  // update parameters
  output.update();
  for (uint32_t i = 0 ; i < hidden.size() ; ++i) {
    hidden[i].update();
  }
}

mat NeuralNet::predict(const mat sample) {
  // prediction is simply forwardprop
  mat current = input.forwardprop(sample);
  for (uint32_t i = 0 ; i < hidden.size() ; ++i) {
    mat n = hidden[i].forwardprop(current);
    current = n;
  }
  result = output.forwardprop(current);
  mat argmax = output.argmax();
  return argmax;
}

void NeuralNet::gradcheck() {
  // back prop result from output to input
#ifdef DEBUG
  const double diff = 1e-6;
  mat w = output.getw();
  mat ngrad = computengrad(w.n_rows, w.n_cols, hidden.size());
  mat grad = output.getgrad();
  cout << "gradient checking ......";
  for (uint32_t i = 0 ; i < grad.n_rows ; ++i) {
    for (uint32_t j = 0 ; j < grad.n_cols ; ++j) {
      if (ngrad(i, j) > grad(i, j)) {
        if (ngrad(i, j) - grad(i, j) >= diff) {
          cout << " failed" << endl;
          cout << grad;
          cout << ngrad;
          return;
        }
      } else {
        if (grad(i, j) - ngrad(i, j) >= diff) {
          cout << " failed" << endl;
          cout << grad;
          cout << ngrad;
          return;
        }
      }
    }
  }
  for (int i = hidden.size()-1 ; i >= 0 ; --i) {
    w = hidden[i].getw();
    ngrad = computengrad(w.n_rows, w.n_cols, i);
    grad = hidden[i].getgrad();

    for (uint32_t i = 0 ; i < grad.n_rows ; ++i) {
      for (uint32_t j = 0 ; j < grad.n_cols ; ++j) {
        if (ngrad(i, j) > grad(i, j)) {
          if (ngrad(i, j) - grad(i, j) >= diff) {
            cout << " failed" << endl;
            cout << grad;
            cout << ngrad;
            return;
          }
        } else {
          if (grad(i, j) - ngrad(i, j) >= diff) {
            cout << " failed" << endl;
            cout << grad;
            cout << ngrad;
            return;
          }
        }
      }
    }
  }
  cout << " passed." << endl;
#endif
}

double NeuralNet::computecost() {
  // forward propagation
  //mat current = input.forwardprop(x);
  //for (uint32_t i = 0 ; i < hidden.size() ; ++i) {
    //mat n = hidden[i].forwardprop(current);
    //current = n;
  //}
  //mat out = output.forwardprop(current);
  mat J = cost(y, result);
  double val = 0;
  for (uint32_t i = 0 ; i < J.n_rows ; ++i) {
    for (uint32_t j = 0 ; j < J.n_cols ; ++j) {
      val += J(i, j);
    }
  }
  return val;
}

mat NeuralNet::getresult() const {
  return result;
}

double NeuralNet::computecost(const mat perturb, const uint32_t idx) {
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
  return val;
}

mat NeuralNet::computengrad(const int nrows, const int ncols, const int idx) {
  mat wgrad(nrows, ncols);
  mat perturb(nrows, ncols);
  wgrad.zeros();
  perturb.zeros();

  for (int i = 0 ; i < nrows ; ++i) {
    for (int j = 0 ; j < ncols ; ++j) {
      perturb(i, j) = eps;
      const double loss1 = computecost(perturb, idx);
      const double loss2 = computecost(-perturb, idx);
      wgrad(i, j) = (loss1 - loss2) / (2 * eps);

      perturb(i, j) = 0;
    }
  }

  return wgrad;
}

}
