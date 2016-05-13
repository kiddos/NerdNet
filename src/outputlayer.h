#ifndef OUTPUTLAYER_H
#define OUTPUTLAYER_H

#include "layer.h"

namespace nn {

class OutputLayer : public Layer {
 public:
  OutputLayer();
  OutputLayer(const OutputLayer &output);
  OutputLayer(const int pnnodes, const int outputnodes,
              const double lrate, const double lambda,
              func act, func actd,
              matfunc cost, matfuncd costd);
  OutputLayer(const int pnnodes, const int outputnodes,
              const double lrate,
              const double lambda,
              const ActFunc actfunc,
              matfunc cost, matfuncd costd);
  virtual OutputLayer& operator= (const OutputLayer &output);
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

/*** Softmax Output ***/
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

} /* end of nn namespace */

#endif /* end of include guard: OUTPUTLAYER_H */
