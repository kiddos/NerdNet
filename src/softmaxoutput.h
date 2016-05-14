#ifndef SOFTMAXOUTPUT_H
#define SOFTMAXOUTPUT_H

#include "outputlayer.h"

namespace nn {

class SoftmaxOutput : public OutputLayer {
 public:
  SoftmaxOutput();
  SoftmaxOutput(const SoftmaxOutput& output);
  SoftmaxOutput(const int pnnodes, const int outputnodes,
                const double lrate, const double lambda);

  static mat costfunc(mat y, mat h);
  static mat costfuncdelta(mat y, mat a, mat z);
};

}

#endif /* end of include guard: SOFTMAXOUTPUT_H */
