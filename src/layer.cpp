#include "layer.h"

namespace nn {

Layer::Layer() : lrate(0), lambda(0) {
  act = [] (double x) {return x;};
  actd = [] (double x) {return (x=1);};
}

Layer::Layer(const Layer& layer)
    : lrate(layer.lrate), lambda(layer.lambda),
      pa(layer.pa), z(layer.z), a(layer.a), delta(layer.delta),
      W(layer.W), grad(layer.grad) {
  act = layer.act;
  actd = layer.actd;
}

Layer::Layer(const int pnnodes, const int nnodes,
             const double lrate, const double lambda,
             func act, func actd)
    : lrate(lrate), lambda(lambda), act(act), actd(actd),
      W(pnnodes+1, nnodes), grad(pnnodes+1, nnodes) {
  randominit(sqrt(pnnodes));
}

Layer::Layer(const int pnnodes, const int nnodes, const double lrate,
             const double lambda, ActFunc actfunc)
    : lrate(lrate), lambda(lambda), act(actfunc.act), actd(actfunc.actd),
      W(pnnodes+1, nnodes), grad(pnnodes+1, nnodes) {
  randominit(sqrt(pnnodes));
}

Layer& Layer::operator= (const Layer& layer) {
  lrate = layer.lrate;
  lambda = layer.lambda;

  act = layer.act;
  actd = layer.actd;

  pa = layer.pa;
  z = layer.z;
  a = layer.a;
  delta = layer.delta;

  W = layer.W;
  grad = layer.grad;
  return *this;
}

void Layer::randominit(const double eps) {
  const int scale = 10000;
  for (uint32_t i = 0 ; i < W.n_rows ; ++i) {
    for (uint32_t j = 0 ; j < W.n_cols ; ++j) {
      W(i, j) = static_cast<double>(rand() % scale) / static_cast<double>(scale);
      W(i, j) = W(i, j) * 2 * eps;
      W(i, j) = W(i, j) - eps;
    }
  }
}

mat Layer::forwardprop(const mat& pa) {
  this->pa = addcol(pa);
  z = this->pa * W;
  a = funcop(z, act);

  return a;
}

mat Layer::backprop(const mat& d) {
  // compute this delta and grad
  mat actdz = funcop(z, actd);
  mat delta = d;
  delta = delta % addcol(actdz);

  delta.shed_col(0);
  grad = pa.t() * delta;
  // regularization
  grad = grad + lambda * W;

  // compute new delta to throw to next layer
  mat newdelta = delta * W.t();
  return newdelta;
}

void Layer::update() {
  W = W - lrate * grad;
}

void Layer::update(const mat grad) {
  W = W - lrate * grad;
}

}
