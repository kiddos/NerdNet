#include "kullbackleibleroutput.h"

namespace nn {

KullbackLeiblerOutput::KullbackLeiblerOutput() {}

KullbackLeiblerOutput::KullbackLeiblerOutput(const KullbackLeiblerOutput& output)
    : OutputLayer(output) {}

KullbackLeiblerOutput::KullbackLeiblerOutput(const int pnnodes, const int outputnodes,
                                             const double lrate, const double lambda)
    : OutputLayer(pnnodes, outputnodes, lrate, lambda, identity,
                  KullbackLeiblerOutput::costfunc,
                  KullbackLeiblerOutput::costfuncdelta) {}

mat KullbackLeiblerOutput::costfunc(mat y, mat h) {
  const mat J = y % logorithm(y / h) / y.n_rows;
  return J;
}

mat KullbackLeiblerOutput::costfuncdelta(mat y, mat a, mat) {
  return (y / a) / y.n_rows;
}

}
