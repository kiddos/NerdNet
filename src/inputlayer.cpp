#include "inputlayer.h"

namespace nn {

InputLayer::InputLayer() {}

InputLayer::InputLayer(const InputLayer& input)
    : Layer(input.getpnnodes(), input.getw().n_cols, input.getlrate(),
            input.getlambda(), input.getact(), input.getactd()) {}

InputLayer::InputLayer(const int innodes) {
  lrate = 0;
  lambda = 0;
  this->act = [](double x) {return x;};
  this->actd = [](double) {return 1.0;};
  W = mat(innodes, innodes);
  grad = mat(innodes, innodes);
}

InputLayer& InputLayer::operator= (const InputLayer& input) {
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

}
