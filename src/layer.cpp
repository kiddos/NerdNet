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

// Layer implementation
Layer::Layer() : nnodes(0), lrate(0) {
  act = [] (double x) {return x;};
  actd = [] (double x) {return (x=1);};
}

Layer::Layer(const Layer &l) : nnodes(l.getnnodes()), lrate(l.getlrate()) {
  z = l.getz();
  a = l.geta();
  W = l.getw();
  grad = l.getgrad();
}

Layer::Layer(const int nnodes, const int nextnnodes, const double lrate,
             double (*act)(double), double (*actd)(double)) :
             nnodes(nnodes+1), lrate(lrate), act(act), actd(actd) {
  W = mat(this->nnodes, nextnnodes);
  grad = mat(this->nnodes, nextnnodes);

#ifndef LIBMAT
  W.randn();
#else
#endif
}

void Layer::operator= (const Layer &l) {
  nnodes = l.getnnodes();
  lrate = l.getlrate();
  act = l.getact();
  actd = l.getactd();
  z = l.getz();
  a = l.geta();
  W = l.getw();
  grad = l.getgrad();
  delta = l.getdelta();
}

mat Layer::forwardprop(const mat pa) {
  z = pa * W;
  a = funcop(z, act);

  return a;
}

mat Layer::backprop(const mat delta) {
  grad = a.t() * delta;
  mat actdz = funcop(z, actd);
  return delta * W.t() % actdz;
}

void Layer::update() {
  W = W - lrate * grad;
}

int Layer::getnnodes() const {
  return nnodes;
}

double Layer::getlrate() const {
  return lrate;
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

// InputLayer implementation
InputLayer::InputLayer() {}

InputLayer::InputLayer(const InputLayer &input) :
    Layer(input.getnnodes(), input.getw().n_cols, input.getlrate(),
          input.getact(), input.getactd()) {}

InputLayer::InputLayer(const int innodes, const double lrate,
                       double (*act)(double), double (*actd)(double)) {
  nnodes = innodes;
  this->lrate = lrate;
  this->act = act;
  this->actd = actd;
  // not using W and grad for input layer
}

void InputLayer::operator= (const InputLayer &input) {
  nnodes = input.getnnodes();
  lrate = input.getlrate();
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

OutputLayer::OutputLayer(const OutputLayer &output) :
    Layer(output.getnnodes(), output.getw().n_cols, output.getlrate(),
          output.getact(), output.getactd()) {
  cost = output.getcost();
  costd = output.getcostd();
}

OutputLayer::OutputLayer(const int nnodes, const int outputnodes, const double lrate,
                         double (*act)(double),
                         double (*actd)(double),
                         mat (*cost)(mat,mat),
                         mat (*costd)(mat,mat,mat)) {
  this->nnodes = nnodes + 1;
  this->lrate = lrate;
  this->act = act;
  this->actd = actd;
  this->cost = cost;
  this->costd = costd;

  W = mat(this->nnodes, outputnodes);
  grad = mat(this->nnodes, outputnodes);

#ifndef LIBMAT
  W.randn();
#else
#endif
}

void OutputLayer::operator= (const OutputLayer &output) {
  nnodes = output.getnnodes();
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

mat OutputLayer::backprop(const mat label) {
  y = label;
  delta = costd(y, a, z);
  return delta;
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
