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
        const double lambda, ActFunc act);
  virtual void operator=(const Layer &l);
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
  virtual void operator=(const InputLayer &input);
  virtual mat forwardprop(const mat input);
};

class OutputLayer : public Layer {
 public:
  OutputLayer();
  OutputLayer(const OutputLayer &output);
  OutputLayer(const int nnodes, const int outputnodes,
              const double lrate,
              const double lambda,
              double (*act)(double),
              double (*actd)(double),
              mat (*cost)(mat,mat),
              mat (*costd)(mat,mat,mat,mat));
  virtual void operator=(const OutputLayer &output);
  virtual mat backprop(const mat label);
  mat argmax() const;
  double getcostval() const;
  mfunc getcost() const;
  mfuncd getcostd() const;

 protected:
  mat (*cost)(mat,mat);
  mat (*costd)(mat,mat,mat,mat);
  mat y;
};

class SoftmaxOutput : public OutputLayer {
 public:
  SoftmaxOutput();
  SoftmaxOutput(const OutputLayer &output);
  SoftmaxOutput(const int nnodes, const int outputnodes,
                const double lrate, const double lambda);

};

class QuadraticOutput : public OutputLayer {
  QuadraticOutput();
  QuadraticOutput(const QuadraticOutput &output);
  QuadraticOutput(const int nnodes, const int outputnodes,
                  const double lrate, const double lambda);
};

class CrossEntropyOutput : public OutputLayer {
  CrossEntropyOutput();
  CrossEntropyOutput(const CrossEntropyOutput &output);
  CrossEntropyOutput(const int nnodes, const int outputnodes,
                     const double lrate, const double lambda);
};

class ExponentialOutput : public OutputLayer {
  ExponentialOutput();
  ExponentialOutput(const CrossEntropyOutput &output);
  ExponentialOutput(const int nnodes, const int outputnodes,
                     const double lrate, const double lambda,
                     const double tou);
};

class KullbackLeiblerOutput : public OutputLayer {
  KullbackLeiblerOutput();
  KullbackLeiblerOutput(const KullbackLeiblerOutput &output);
  KullbackLeiblerOutput(const int nnodes, const int outputnodes,
                        const double lrate, const double lambda);
};

}

#endif /* end of include guard: LAYER_H */
