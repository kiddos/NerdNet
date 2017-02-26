#ifndef QUADRATICOUTPUT_H
#define QUADRATICOUTPUT_H

#include "outputlayer.h"

namespace nn {

class QuadraticOutput : public OutputLayer {
 public:
  QuadraticOutput();
  QuadraticOutput(const QuadraticOutput& output);
  QuadraticOutput(const int pnnodes, const int outputnodes,
                  const double lrate, const double lambda);
  QuadraticOutput(const int pnnodes, const int outputnodes,
                  const double lrate, const double stddev,
                  const double lambda);
  QuadraticOutput(LayerParam param);

  static mat costfunc(mat y, mat h);
  static mat costfuncdelta(mat y, mat a, mat z);
};

}

#endif /* end of include guard: QUADRATICOUTPUT_H */
