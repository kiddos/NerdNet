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
Layer::Layer() : lrate(0), lambda(0) {
  act = [] (double x) {return x;};
  actd = [] (double x) {return (x=1);};
}

Layer::Layer(const Layer &l)
    : lrate(l.getlrate()), lambda(l.getlambda()),
      z(l.getz()), a(l.geta()), W(l.getw()), grad(l.getgrad()) {
  act = l.getact();
  actd = l.getactd();
}

Layer::Layer(const int pnnodes, const int nnodes, const double lrate,
             const double lambda, double (*act)(double), double (*actd)(double))
    : lrate(lrate), lambda(lambda), act(act), actd(actd),
      W(pnnodes+1, nnodes), grad(pnnodes+1, nnodes) {
  randominit(6.0);
}

Layer::Layer(const int pnnodes, const int nnodes, const double lrate,
             const double lambda, ActFunc actfunc)
    : lrate(lrate), lambda(lambda), act(actfunc.act), actd(actfunc.actd),
      W(pnnodes+1, nnodes), grad(pnnodes+1, nnodes) {
  randominit(6.0);
}

Layer& Layer::operator= (const Layer &l) {
  lrate = l.getlrate();
  lambda = l.getlambda();
  act = l.getact();
  actd = l.getactd();
  z = l.getz();
  a = l.geta();
  W = l.getw();
  grad = l.getgrad();
  delta = l.getdelta();
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

// InputLayer implementation
InputLayer::InputLayer() {}

InputLayer::InputLayer(const InputLayer &input)
    : Layer(input.getpnnodes(), input.getw().n_cols, input.getlrate(),
            input.getlambda(), input.getact(), input.getactd()) {}

InputLayer::InputLayer(const int innodes) {
  lrate = 0;
  lambda = 0;
  this->act = [](double x) {return x;};
  this->actd = [](double) {return 1.0;};
  W = mat(innodes, innodes);
  grad = mat(innodes, innodes);
  // not using W and grad for input layer
  // TODO implement rbf network
}

InputLayer& InputLayer::operator= (const InputLayer &input) {
  lrate = input.getlrate();
  lambda = input.getlambda();

  z = input.getz();
  a = input.geta();
  W = input.getw();
  grad = input.getgrad();
  delta = input.getdelta();

  act = input.getact();
  actd = input.getactd();

  return *this;
}

mat InputLayer::forwardprop(const mat input) {
  z = input;
  a = input;
  return a;
}

// OutputLayer implementation
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

/*** Softmax Regression ***/
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

