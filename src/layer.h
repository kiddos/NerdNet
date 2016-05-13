#ifndef LAYER_H
#define LAYER_H

#include <stdint.h>
#include "type.h"
#include "act.h"

namespace nn {

mat funcop(const mat m, double (*f)(double));
mat addcol(const mat m, const double val);

class Layer {
 public:
  Layer();
  Layer(const Layer &l);
  Layer(const int pnnodes, const int nnodes, const double lrate,
        const double lambda, double (*act)(double), double (*actd)(double));
  Layer(const int pnnodes, const int nnodes, const double lrate,
        const double lambda, ActFunc actfunc);
  virtual Layer& operator= (const Layer &l);
  virtual mat forwardprop(const mat pa);
  virtual mat backprop(const mat delta);
  virtual void update();
  void randominit(const double eps);

  int getpnnodes() const;
  int getnnodes() const;
  double getlrate() const;
  double getlambda() const;
  mat getz() const;
  mat geta() const;
  mat getw() const;
  mat getgrad() const;
  mat getdelta() const;
  func getact() const;
  func getactd() const;
  void setw(mat w);
  void setlrate(const double lrate);
  void setlambda(const double lambda);

 protected:
  double lrate;
  double lambda;
  unsigned long long int iters;
  func act, actd;
  mat pa, z, a;
  mat W, grad, delta, momentum;
};

}

#endif /* end of include guard: LAYER_H */
