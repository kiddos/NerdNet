#ifndef LAYER_H
#define LAYER_H

#include <stdint.h>
#include "matop.h"
#include "type.h"
#include "act.h"

namespace nn {

class Layer {
 public:
  Layer();
  Layer(const Layer& layer);
  Layer(const int pnnodes, const int nnodes,
        const double lrate, const double lambda,
        func act, func actd);
  Layer(const int pnnodes, const int nnodes,
        const double lrate, const double lambda,
        ActFunc actfunc);
  virtual Layer& operator= (const Layer& layer);
  virtual mat forwardprop(const mat pa);
  virtual mat backprop(const mat delta);
  virtual void update();
  virtual void update(const mat grad);
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
  func act, actd;
  mat pa, z, a, delta;
  mat W, grad;
};

}

#endif /* end of include guard: LAYER_H */
