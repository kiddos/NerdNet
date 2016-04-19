#include "layer.h"

namespace nn {

mat funcop(const mat m, double (*f)(double)) {
  mat newmat(m.n_rows, m.n_cols);
  for (uint32_t i = 0 ; i < m.n_rows ; ++i) {
    for (uint32_t j = 0 ; j < m.n_cols ; ++j) {
      newmat(i, j) = f(m(i, j));
    }
  }
  return newmat;
}

mat addcol(const mat m, const double val) {
  mat newmat(m.n_rows, m.n_cols+1);
  for (uint32_t i = 0 ; i < m.n_rows ; ++i) {
    newmat(i, 0) = val;
    for (uint32_t j = 0 ; j < m.n_cols ; ++j) {
      newmat(i, j+1) = m(i, j);
    }
  }
  return newmat;
}

// Layer implementation
Layer::Layer() : pnnodes(0), lrate(0), lambda(0) {
  act = [] (double x) {return x;};
  actd = [] (double x) {return (x=1);};
}

Layer::Layer(const Layer &l) :
    pnnodes(l.getpnnodes()), lrate(l.getlrate()), lambda(l.getlambda()) {
  z = l.getz();
  a = l.geta();
  W = l.getw();
  grad = l.getgrad();
  act = l.getact();
  actd = l.getactd();
}

Layer::Layer(const int pnnodes, const int nnodes, const double lrate,
             const double lambda, double (*act)(double), double (*actd)(double)) :
             pnnodes(pnnodes+1), lrate(lrate), lambda(lambda), act(act), actd(actd) {
  W = mat(this->pnnodes, nnodes);
  grad = mat(this->pnnodes, nnodes);

#ifndef LIBMAT
  W.imbue([] () {return rand() % 10000 / 10000.0;});
#else
#endif
}

void Layer::operator= (const Layer &l) {
  pnnodes = l.getpnnodes();
  lrate = l.getlrate();
  lambda = l.getlambda();
  act = l.getact();
  actd = l.getactd();
  z = l.getz();
  a = l.geta();
  W = l.getw();
  grad = l.getgrad();
  delta = l.getdelta();
}

mat Layer::forwardprop(const mat pa) {
  this->pa = addcol(pa, 1);
  z = this->pa * W;
  a = funcop(z, act);

  return a;
}

mat Layer::backprop(const mat d) {
  // compute this delta and grad
  mat actdz = funcop(z, actd);
  mat delta = d;
  delta = delta % addcol(actdz, 1);

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

int Layer::getpnnodes() const {
  return pnnodes;
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

// InputLayer implementation
InputLayer::InputLayer() {}

InputLayer::InputLayer(const InputLayer &input) :
    Layer(input.getpnnodes(), input.getw().n_cols, input.getlrate(),
          input.getlambda(), input.getact(), input.getactd()) {}

InputLayer::InputLayer(const int innodes) {
  pnnodes = innodes;
  lrate = 1;
  this->act = [](double x) {return x;};
  this->actd = [](double x) {return (x=1);};
  // not using W and grad for input layer
}

void InputLayer::operator= (const InputLayer &input) {
  pnnodes = input.getpnnodes();
  lrate = input.getlrate();
  lambda = input.getlambda();
  act = input.getact();
  actd = input.getactd();
  z = input.getz();
  a = input.geta();
  W = input.getw();
  grad = input.getgrad();
  delta = input.getdelta();
}

mat InputLayer::forwardprop(const mat input) {
  z = input;
  a = input;
  return a;
}

// OutputLayer implementation
OutputLayer::OutputLayer() {}

OutputLayer::OutputLayer(const OutputLayer &output) {
  pnnodes = output.getpnnodes();
  lrate = output.getlrate();
  act = output.getact();
  actd = output.getactd();
  z = output.getz();
  a = output.geta();
  W = output.getw();
  grad = output.getgrad();
  delta = output.getdelta();
  cost = output.getcost();
  costd = output.getcostd();
}

OutputLayer::OutputLayer(const int pnnodes, const int outputnodes,
                         const double lrate,
                         double (*act)(double),
                         double (*actd)(double),
                         mat (*cost)(mat,mat),
                         mat (*costd)(mat,mat,mat,mat)) {
  this->pnnodes = pnnodes + 1;
  this->lrate = lrate;
  this->act = act;
  this->actd = actd;
  this->cost = cost;
  this->costd = costd;

  W = mat(this->pnnodes, outputnodes);
  grad = mat(this->pnnodes, outputnodes);

#ifndef LIBMAT
  W.randn();
#else
#endif
}

void OutputLayer::operator= (const OutputLayer &output) {
  pnnodes = output.getpnnodes();
  lrate = output.getlrate();
  lambda = output.getlambda();
  act = output.getact();
  actd = output.getactd();
  z = output.getz();
  a = output.geta();
  W = output.getw();
  grad = output.getgrad();
  delta = output.getdelta();
  cost = output.getcost();
  costd = output.getcostd();
}

mat OutputLayer::backprop(const mat label) {
  y = label;
  delta = costd(y, a, z, pa);
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

mfunc OutputLayer::getcost() const {
  return cost;
}

mfuncd OutputLayer::getcostd() const {
  return costd;
}

}
