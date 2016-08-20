#ifndef LAYER_H
#define LAYER_H

#include <stdint.h>
#include "matop.h"
#include "type.h"
#include "act.h"
#include "config.h"

namespace nn {

struct LayerParam {
  LayerParam() {
    previous_nodes = nodes = 0;
    standard_dev = 1.0;
    learning_rate = 1e-2;
    lambda = 0;
    actfunc = identity;
  }
  int previous_nodes, nodes;
  double standard_dev, learning_rate, lambda;
  ActFunc actfunc;
};

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
  virtual mat forwardprop(const mat& pa);
  virtual mat backprop(const mat& delta);
  virtual void update();
  virtual void update(const mat grad);
  virtual void randominit(const double eps);

  int getpnnodes() const { return W.n_rows; };
  int getnnodes() const { return W.n_cols; };
  double getlrate() const { return lrate; };
  double getlambda() const { return lambda; };
  mat getz() const { return z; };
  mat geta() const { return a; };
  mat getw() const { return W; };
  mat getgrad() const { return grad; };
  mat getdelta() const {return delta; };
  func getact() const { return act; };
  func getactd() const { return actd; };
  void setw(mat w) { W = w; };
  void setlrate(const double lrate) { this->lrate = lrate; };
  void setlambda(const double lambda) { this->lambda = lambda; };

 protected:
  double lrate;
  double lambda;
  func act, actd;
  mat pa, z, a, delta;
  mat W, grad;
};

}

#endif /* end of include guard: LAYER_H */
