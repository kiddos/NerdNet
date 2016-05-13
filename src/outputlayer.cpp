#include "outputlayer.h"

namespace nn {

OutputLayer::OutputLayer() {}

OutputLayer::OutputLayer(const OutputLayer &output) {
  lrate = output.getlrate();
  lambda = output.getlambda();

  // abandone previous node input pa which is not crucial
  z = output.getz();
  a = output.geta();
  W = output.getw();
  grad = output.getgrad();
  delta = output.getdelta();

  act = output.getact();
  actd = output.getactd();

  cost = output.getcost();
  costd = output.getcostd();
}

OutputLayer::OutputLayer(const int pnnodes, const int outputnodes,
                         const double lrate, const double lambda,
                         double (*act)(double), double (*actd)(double),
                         matfunc cost, matfuncd costd)
    : Layer(pnnodes, outputnodes, lrate, lambda, act, actd) {
  this->cost = cost;
  this->costd = costd;
}

OutputLayer::OutputLayer(const int pnnodes, const int outputnodes,
                         const double lrate, const double lambda,
                         const ActFunc actfunc,
                         matfunc cost, matfuncd costd)
    : Layer(pnnodes, outputnodes, lrate, lambda, actfunc) {
  this->cost = cost;
  this->costd = costd;
}

OutputLayer& OutputLayer::operator= (const OutputLayer &output) {
  lrate = output.getlrate();
  lambda = output.getlambda();

  // abadone pa
  z = output.getz();
  a = output.geta();
  W = output.getw();
  grad = output.getgrad();
  delta = output.getdelta();

  act = output.getact();
  actd = output.getactd();

  cost = output.getcost();
  costd = output.getcostd();

  return *this;
}

mat OutputLayer::backprop(const mat label) {
  y = label;
  delta = costd(y, a, z);
  grad = pa.t() * delta;
  // regularization
  grad = grad + lambda * W;
  // compute next layer delta
  mat newdelta = delta * W.t();
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
  double val = 0;
  for (uint32_t i = 0 ; i < J.n_rows ; ++i) {
    for (uint32_t j = 0 ; j < J.n_cols ; ++j) {
      val += J(i, j);
    }
  }
  return val;
}

matfunc OutputLayer::getcost() const {
  return cost;
}

matfuncd OutputLayer::getcostd() const {
  return costd;
}

/*** Softmax Output ***/
SoftmaxOutput::SoftmaxOutput() {}

SoftmaxOutput::SoftmaxOutput(const SoftmaxOutput &output)
    : OutputLayer(output) {}

SoftmaxOutput::SoftmaxOutput(const int pnnodes, const int outputnodes,
                             const double lrate, const double lambda)
    : OutputLayer(pnnodes, outputnodes, lrate, lambda, identity,
                  SoftmaxOutput::costfunc,
                  SoftmaxOutput::costfuncdelta) {}

mat SoftmaxOutput::costfunc(mat y, mat h) {
  const mat expo = arma::exp(h);
  const mat sumexpo = arma::repmat(arma::sum(expo, 1), 1, y.n_cols);
  const mat P = expo % (1.0 / sumexpo);
  const mat J = - (y % arma::log(P));
  return J;
}

mat SoftmaxOutput::costfuncdelta(mat y, mat a, mat) {
  const mat expo = arma::exp(a);
  const mat sumexpo = arma::repmat(arma::sum(expo, 1), 1, y.n_cols);
  const mat P = expo % (1.0 / sumexpo);
  const mat delta = P - y;
  return delta;
}


/*** Quadratic Output (Mean Square Error) ***/
QuadraticOutput::QuadraticOutput() {}

QuadraticOutput::QuadraticOutput(const QuadraticOutput &output)
    : OutputLayer(output) {}

QuadraticOutput::QuadraticOutput(const int pnnodes, const int outputnodes,
                                 const double lrate, const double lambda)
    : OutputLayer(pnnodes, outputnodes, lrate, lambda, identity,
                  QuadraticOutput::costfunc,
                  QuadraticOutput::costfuncdelta) {}

mat QuadraticOutput::costfunc(mat y, mat h) {
  const mat diff = y - h;
  const mat J = (diff % diff) / 2.0;
  return J;
}

mat QuadraticOutput::costfuncdelta(mat y, mat a, mat) {
  return a - y;
}

/*** Cross Entropy output ***/
CrossEntropyOutput::CrossEntropyOutput() {}

CrossEntropyOutput::CrossEntropyOutput(const CrossEntropyOutput &output)
    : OutputLayer(output) {}

CrossEntropyOutput::CrossEntropyOutput(const int pnnodes, const int outputnodes,
                                       const double lrate, const double lambda)
    : OutputLayer(pnnodes, outputnodes, lrate, lambda, sigmoid,
                  CrossEntropyOutput::costfunc,
                  CrossEntropyOutput::costfuncdelta) {}

mat CrossEntropyOutput::costfunc(mat y, mat h) {
  const mat J = -(y % arma::log(h) + (1-y) % arma::log(1-h));
  return J;
}

mat CrossEntropyOutput::costfuncdelta(mat y, mat a, mat) {
  return a - y;
}

/*** Kullback-Leibler output ***/
KullbackLeiblerOutput::KullbackLeiblerOutput() {}

KullbackLeiblerOutput::KullbackLeiblerOutput(const KullbackLeiblerOutput &output)
    : OutputLayer(output) {}

KullbackLeiblerOutput::KullbackLeiblerOutput(const int pnnodes, const int outputnodes,
                                             const double lrate, const double lambda)
    : OutputLayer(pnnodes, outputnodes, lrate, lambda, identity,
                  CrossEntropyOutput::costfunc,
                  CrossEntropyOutput::costfuncdelta) {}

mat KullbackLeiblerOutput::costfunc(mat y, mat h) {
  const mat J = y % arma::log(y / h);
  return J;
}

mat KullbackLeiblerOutput::costfuncdelta(mat y, mat a, mat) {
  return y / a;
}

}

