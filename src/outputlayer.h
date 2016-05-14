#ifndef OUTPUTLAYER_H
#define OUTPUTLAYER_H

#include "layer.h"

namespace nn {

class OutputLayer : public Layer {
 public:
  OutputLayer();
  OutputLayer(const OutputLayer& output);
  OutputLayer(const int pnnodes, const int outputnodes,
              const double lrate, const double lambda,
              func act, func actd,
              matfunc cost, matfuncd costd);
  OutputLayer(const int pnnodes, const int outputnodes,
              const double lrate,
              const double lambda,
              const ActFunc actfunc,
              matfunc cost, matfuncd costd);
  virtual OutputLayer& operator= (const OutputLayer& output);
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

} /* end of nn namespace */

#endif /* end of include guard: OUTPUTLAYER_H */
