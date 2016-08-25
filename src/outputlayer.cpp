#include "outputlayer.h"

namespace nn {

OutputLayer::OutputLayer() {}

OutputLayer::OutputLayer(const OutputLayer& output)
    : Layer(output) {
  cost = output.getcost();
  costd = output.getcostd();
}

OutputLayer::OutputLayer(const int pnnodes, const int outputnodes,
                         const double lrate, const double lambda,
                         const ActFunc actfunc,
                         matfunc cost, matfuncd costd)
    : Layer(pnnodes, outputnodes, lrate, lambda, actfunc) {
  this->cost = cost;
  this->costd = costd;
}

OutputLayer::OutputLayer(const int pnnodes, const int outputnodes,
                         const double lrate, const double stddev,
                         const double lambda, const ActFunc actfunc,
                         matfunc cost, matfuncd costd)
    : Layer(pnnodes, outputnodes, lrate, stddev, lambda, actfunc) {
  this->cost = cost;
  this->costd = costd;
}

OutputLayer::OutputLayer(LayerParam param, matfunc cost, matfuncd costd)
    : Layer(param) {
  this->cost = cost;
  this->costd = costd;
}

OutputLayer& OutputLayer::operator= (const OutputLayer& output) {
  Layer::operator= (output);

  cost = output.getcost();
  costd = output.getcostd();
  return *this;
}

mat OutputLayer::backprop(const mat& label) {
  y = label;
  delta = costd(y, a, z);
  grad = pa.t() * delta;
  // regularization
  grad = grad + lambda * W;
  // compute next layer delta
  mat newdelta = delta * W.t();
  newdelta.shed_col(0);
  return newdelta;
}

mat OutputLayer::argmax() const {
  mat result(a.n_rows, 2);
  for (uint32_t i = 0 ; i < a.n_rows ; ++i) {
    double maxval = a(i, 0);
    int maxidx = 0;
    for (uint32_t j = 1 ; j < a.n_cols ; ++j) {
      if (a(i, j) > maxval) {
        maxval = a(i, j);
        maxidx = j;
      }
    }
    result(i, 0) = maxidx;
    result(i, 1) = maxval;
  }
  return result;
}

double OutputLayer::getcostval() const {
  mat J = cost(y, a);
  return sumall(J);
}

matfunc OutputLayer::getcost() const {
  return cost;
}

matfuncd OutputLayer::getcostd() const {
  return costd;
}

}
