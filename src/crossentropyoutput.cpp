#include "crossentropyoutput.h"

namespace nn {

CrossEntropyOutput::CrossEntropyOutput() {}

CrossEntropyOutput::CrossEntropyOutput(const CrossEntropyOutput& output)
    : OutputLayer(output) {}

CrossEntropyOutput::CrossEntropyOutput(const int pnnodes, const int outputnodes,
                                       const double lrate, const double lambda)
    : OutputLayer(pnnodes, outputnodes, lrate, lambda, sigmoid,
                  CrossEntropyOutput::costfunc,
                  CrossEntropyOutput::costfuncdelta) {}

mat CrossEntropyOutput::costfunc(mat y, mat h) {
  const mat J = -(y % arma::log(h) + (1-y) % arma::log(1-h));
  return J;
}

mat CrossEntropyOutput::costfuncdelta(mat y, mat a, mat) {
  return a - y;
}

}
