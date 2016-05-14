#ifndef CROSSENTROPYOUTPUT_H
#define CROSSENTROPYOUTPUT_H

#include "outputlayer.h"

namespace nn {

class CrossEntropyOutput : public OutputLayer {
 public:
  CrossEntropyOutput();
  CrossEntropyOutput(const CrossEntropyOutput& output);
  CrossEntropyOutput(const int pnnodes, const int outputnodes,
                     const double lrate, const double lambda);

  static mat costfunc(mat y, mat h);
  static mat costfuncdelta(mat y, mat a, mat z);
};

}

#endif /* end of include guard: CROSSENTROPYOUTPUT_H */
