#ifndef KULLBACKLEIBLEROUTPUT_H
#define KULLBACKLEIBLEROUTPUT_H

#include "outputlayer.h"

namespace nn {

class KullbackLeiblerOutput : public OutputLayer {
 public:
  KullbackLeiblerOutput();
  KullbackLeiblerOutput(const KullbackLeiblerOutput& output);
  KullbackLeiblerOutput(const int pnnodes, const int outputnodes,
                        const double lrate, const double lambda);
  KullbackLeiblerOutput(const int pnnodes, const int outputnodes,
                        const double lrate, const double stddev,
                        const double lambda);
  KullbackLeiblerOutput(LayerParam);

  static mat costfunc(mat y, mat h);
  static mat costfuncdelta(mat y, mat a, mat z);
};

}

#endif /* end of include guard: KULLBACKLEIBLEROUTPUT_H */
