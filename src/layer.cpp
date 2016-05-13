#include "layer.h"

namespace nn {

Layer::Layer() : lrate(0), lambda(0), iters(0), usemomentum(false) {
  act = [] (double x) {return x;};
  actd = [] (double x) {return (x=1);};
}

Layer::Layer(const Layer& layer)
    : lrate(layer.lrate), lambda(layer.lambda),
      iters(layer.iters), usemomentum(layer.usemomentum),
      pa(layer.pa), z(layer.z), a(layer.a), delta(layer.delta),
      W(layer.W), grad(layer.grad), momentum(layer.momentum) {
  act = layer.act;
  actd = layer.actd;
}

Layer::Layer(const int pnnodes, const int nnodes,
             const double lrate, const double lambda,
             double (*act)(double), double (*actd)(double),
             bool usemomentum)
    : lrate(lrate), lambda(lambda),
      iters(0), usemomentum(usemomentum),
      act(act), actd(actd),
      W(pnnodes+1, nnodes), grad(pnnodes+1, nnodes), momentum(pnnodes+1, nnodes) {
  randominit(6.0);
  momentum.zeros();
}

Layer::Layer(const int pnnodes, const int nnodes, const double lrate,
             const double lambda, ActFunc actfunc,
             bool usemomentum)
    : lrate(lrate), lambda(lambda),
      iters(0), usemomentum(usemomentum),
      act(actfunc.act), actd(actfunc.actd),
      W(pnnodes+1, nnodes), grad(pnnodes+1, nnodes), momentum(pnnodes+1, nnodes) {
  randominit(6.0);
  momentum.zeros();
}

Layer& Layer::operator= (const Layer& layer) {
  lrate = layer.lrate;
  lambda = layer.lambda;
  iters = layer.iters;
  usemomentum = layer.usemomentum;

  act = layer.act;
  actd = layer.actd;

  pa = layer.pa;
  z = layer.z;
  a = layer.a;
  delta = layer.delta;

  W = layer.W;
  grad = layer.grad;
  momentum = layer.momentum;
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

mat Layer::forwardprop(const mat pa) {
  this->pa = addcol(pa);
  z = this->pa * W;
  a = funcop(z, act);

  return a;
}

mat Layer::backprop(const mat d) {
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
  if (usemomentum) {
    iters ++;
    momentum = ((iters-1)/iters) * momentum + (1.0/iters) * grad;
    W = W - lrate * momentum;
  } else {
    W = W - lrate * grad;
  }
}

int Layer::getpnnodes() const {
  return W.n_rows;
}

int Layer::getnnodes() const {
  return W.n_cols;
}

double Layer::getlrate() const {
  return lrate;
}

double Layer::getlambda() const {
  return lambda;
}

mat Layer::getz() const {
  return z;
}

mat Layer::geta() const {
  return a;
}

mat Layer::getw() const {
  return W;
}

mat Layer::getgrad() const {
  return grad;
}

mat Layer::getdelta() const {
  return delta;
}

func Layer::getact() const {
  return act;
}

func Layer::getactd() const {
  return actd;
}

void Layer::setw(const mat w) {
  this->W = w;
}

void Layer::setlrate(const double lrate) {
  this->lrate = lrate;
}

void Layer::setlambda(const double lambda) {
  this->lambda = lambda;
}

}

