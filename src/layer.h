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
  virtual Layer& operator=(const Layer &l);
  virtual mat forwardprop(const mat pa);
  virtual mat backprop(const mat delta);
  virtual void update();
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
  void randominit(const double eps);
  double lrate;
  double lambda;
  double (*act)(double);
  double (*actd)(double);
  mat pa, z, a;
  mat W, grad, delta;
};

class InputLayer : public Layer {
 public:
  InputLayer();
  InputLayer(const InputLayer &input);
  InputLayer(const int innodes);
  virtual InputLayer& operator=(const InputLayer &input);
  virtual mat forwardprop(const mat input);
};

class OutputLayer : public Layer {
 public:
  OutputLayer();
  OutputLayer(const OutputLayer &output);
  OutputLayer(const int pnnodes, const int outputnodes,
              const double lrate,
              const double lambda,
              double (*act)(double),
              double (*actd)(double),
              matfunc cost,
              matfuncd costd);
  OutputLayer(const int pnnodes, const int outputnodes,
              const double lrate,
              const double lambda,
              const ActFunc actfunc,
              matfunc cost,
              matfuncd costd);
  virtual OutputLayer& operator=(const OutputLayer &output);
  virtual mat backprop(const mat label);
  mat argmax() const;
  double getcostval() const;
  matfunc getcost() const;
  matfuncd getcostd() const;

 protected:
  matfunc cost;
  matfuncd costd;
  mat y;
};

class SoftmaxOutput : public OutputLayer {
 public:
  SoftmaxOutput();
  SoftmaxOutput(const SoftmaxOutput &output);
  SoftmaxOutput(const int pnnodes, const int outputnodes,
                const double lrate, const double lambda);

  static mat costfunc(mat y, mat h);
  static mat costfuncdelta(mat y, mat a, mat z);
};

class QuadraticOutput : public OutputLayer {
 public:
  QuadraticOutput();
  QuadraticOutput(const QuadraticOutput &output);
  QuadraticOutput(const int pnnodes, const int outputnodes,
                  const double lrate, const double lambda);

  static mat costfunc(mat y, mat h);
  static mat costfuncdelta(mat y, mat a, mat z);
};

class CrossEntropyOutput : public OutputLayer {
 public:
  CrossEntropyOutput();
  CrossEntropyOutput(const CrossEntropyOutput &output);
  CrossEntropyOutput(const int pnnodes, const int outputnodes,
                     const double lrate, const double lambda);

  static mat costfunc(mat y, mat h);
  static mat costfuncdelta(mat y, mat a, mat z);
};

class KullbackLeiblerOutput : public OutputLayer {
 public:
  KullbackLeiblerOutput();
  KullbackLeiblerOutput(const KullbackLeiblerOutput &output);
  KullbackLeiblerOutput(const int pnnodes, const int outputnodes,
                        const double lrate, const double lambda);

  static mat costfunc(mat y, mat h);
  static mat costfuncdelta(mat y, mat a, mat z);
};

}

#endif /* end of include guard: LAYER_H */
