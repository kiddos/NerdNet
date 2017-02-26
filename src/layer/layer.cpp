#include <random>

#include "layer.h"

namespace nn {

Layer::Layer() : lrate(0), lambda(0) {
  actfunc = nn::identity;
}

Layer::Layer(const Layer& layer)
    : lrate(layer.lrate), lambda(layer.lambda),
      pa(layer.pa), z(layer.z), a(layer.a), delta(layer.delta),
      W(layer.W), grad(layer.grad) {
  actfunc = layer.actfunc;
}

Layer::Layer(const int pnnodes, const int nnodes, const double lrate,
             const double lambda, ActFunc actfunc)
    : lrate(lrate), lambda(lambda), actfunc(actfunc),
      W(pnnodes+1, nnodes), grad(pnnodes+1, nnodes) {
  randominit(1.0);
}

Layer::Layer(const int pnnodes, const int nnodes,
             const double lrate, const double standard_dev,
             const double lambda, ActFunc actfunc)
    : lrate(lrate), lambda(lambda), actfunc(actfunc),
      W(pnnodes+1, nnodes), grad(pnnodes+1, nnodes) {
  randominit(standard_dev);
}

Layer::Layer(const LayerParam param)
    : Layer(param.previous_nodes, param.nodes, param.learning_rate,
            param.standard_dev, param.lambda, param.actfunc) {}

Layer& Layer::operator= (const Layer& layer) {
  lrate = layer.lrate;
  lambda = layer.lambda;

  actfunc = layer.actfunc;

  pa = layer.pa;
  z = layer.z;
  a = layer.a;
  delta = layer.delta;

  W = layer.W;
  grad = layer.grad;
  return *this;
}

void Layer::randominit(const double stddev) {
  std::default_random_engine gen;
  std::uniform_real_distribution<double> dist(-stddev, stddev);

  double mean = 0;
  for (int i = 0 ; i < static_cast<int>(W.n_rows) ; ++i) {
    for (int j = 0 ; j < static_cast<int>(W.n_cols) ; ++j) {
      W(i, j) = dist(gen);
      mean += W(i, j);
    }
  }

  double dev = 0;
  for (int i = 0 ; i < static_cast<int>(W.n_rows) ; ++i) {
    for (int j = 0 ; j < static_cast<int>(W.n_cols) ; ++j) {
      const double diff = mean - W(i, j);
      dev += diff * diff;
    }
  }
  dev = sqrt(dev / (W.n_rows * W.n_cols));
  W *= (stddev / dev);
}

mat Layer::forwardprop(const mat& pa) {
  this->pa = addcol(pa);
  z = this->pa * W;
  a = funcop(z, actfunc.act);

  return a;
}

mat Layer::backprop(const mat& d) {
  // compute this delta and grad
  delta = d % funcop(z, actfunc.actd);
  grad = pa.t() * delta;
  // regularization
  grad = grad + lambda * W;

  // compute new delta to throw to next layer
  return mat(delta * W.t()).submat(0, 1, delta.n_rows-1, W.n_rows-1);
}

void Layer::update() {
  W = W - lrate * grad;
}

void Layer::update(const mat grad) {
  W = W - lrate * grad;
}

}
