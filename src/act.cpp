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
  const double tanhyper = tanh(z);
  return 1.0 - tanhyper * tanhyper;
}

static inline double _bipolarsigmoid(double z) {
  const double expo = exp(-z);
  return (1.0 - expo) / (1.0 + expo);
}
static inline double _bipolarsigmoidgrad(double z) {
  const double expo = exp(-z);
  return 2.0 * expo / (1 + expo) / (1 + expo);
}

static inline double _comploglog(double z) {
  return 1.0 - exp(-exp(z));
}
static inline double _complogloggrad(double z) {
  const double expo = exp(-z);
  return - expo * exp(-expo);
}

static inline double _lecuntanh(double z) {
  return 1.7159 * tanh(2.0/3.0*z);
}
static inline double _lecuntanhgrad(double z) {
  return 1.143933 / (1.0 - 4.0/9.0*z*z);
}

static inline double _hardtanh(double z) {
  return fmax(-1, fmin(1, z));
}
static inline double _hardtanhgrad(double z) {
  if (z >= -1 && z <= 1) return 1;
  return 0;
}

static inline double _absolute(double z) {
  return fabs(z);
}
static inline double _absolutegrad(double z) {
  if (z < 0) return -1;
  return 1;
}

static inline double _relu(double z) {
  return z >= 0 ? z : 0;
}
static inline double _relugrad(double z) {
  return z >= 0 ? 1.0 : 0;
}

static inline double _relucos(double z) {
  return fmax(0, z) + cos(z);
}
static inline double _relucosgrad(double z) {
  return z >= 0 ? (1.0 - sin(z)) : -sin(z);
}

static inline double _relusin(double z) {
  return fmax(0, z) + sin(z);
}
static inline double _relusingrad(double z) {
  return z >= 0 ? (1.0 + cos(z)) : cos(z);
}

static inline double _smoothrelu(double z) {
  return log(1.0 + exp(z));
}
static inline double _smoothrelugrad(double z) {
  const double expo = exp(z);
  return expo / (1.0 + expo);
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
ActFunc comploglog = initactfunc(_comploglog, _complogloggrad);
ActFunc tanhyperbolic = initactfunc(_tanh, _tanhgrad);
ActFunc bipolarsigmoid = initactfunc(_bipolarsigmoid, _bipolarsigmoidgrad);
ActFunc lecuntanh = initactfunc(_lecuntanh, _lecuntanhgrad);
ActFunc hardtanh = initactfunc(_hardtanh, _hardtanhgrad);
ActFunc absolute = initactfunc(_absolute, _absolutegrad);
ActFunc relu = initactfunc(_relu, _relugrad);
ActFunc relucos = initactfunc(_relucos, _relucosgrad);
ActFunc relusin = initactfunc(_relusin, _relusingrad);
ActFunc smoothrelu = initactfunc(_smoothrelu, _smoothrelugrad);

}


