#include "act.h"

namespace nn {

static inline double _identity(double z) {
  return z;
}
static inline double _identitygrad(double) {
  return 1.0;
}

static inline double _sigmoid(double z) {
  return 1.0 / (1 + exp(-z));
}
static inline double _sigmoidgrad(double z) {
  const double expo = exp(-z);
  return expo / (1 + expo) / (1 + expo);
}

static inline double _arctan(double z) {
  return atan(z);
}
static inline double _arctangrad(double z) {
  return 1.0 / (1.0 + z*z);
}

static inline double _tanh(double z) {
  return tanh(z);
}
static inline double _tanhgrad(double z) {
  return 1.0 / (1.0 - z*z);
}

static inline double _bipolarsigmoid(double z) {
  const double expo = exp(-z);
  return (1.0 - expo) / (1.0 + expo);
}
static inline double _bipolarsigmoidgrad(double z) {
  const double expo = exp(-z);
  return 2.0 * expo / (1 + expo) / (1 + expo);
}

ActFunc initactfunc(double (*act)(double), double (*actd)(double)) {
  ActFunc func;
  func.act = act;
  func.actd = actd;
  return func;
}

ActFunc identity = initactfunc(_identity, _identitygrad);
ActFunc sigmoid = initactfunc(_sigmoid, _sigmoidgrad);
ActFunc arctan = initactfunc(_arctan, _arctangrad);
ActFunc tanhyperbolic = initactfunc(_tanh, _tanhgrad);
ActFunc bipolarsigmoid = initactfunc(_bipolarsigmoid, _bipolarsigmoidgrad);

}


