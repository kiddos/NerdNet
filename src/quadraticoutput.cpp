#include "quadraticoutput.h"

namespace nn {

QuadraticOutput::QuadraticOutput() {}

QuadraticOutput::QuadraticOutput(const QuadraticOutput& output)
    : OutputLayer(output) {}

QuadraticOutput::QuadraticOutput(const int pnnodes, const int outputnodes,
                                 const double lrate, const double lambda)
    : OutputLayer(pnnodes, outputnodes, lrate, lambda, identity,
                  QuadraticOutput::costfunc,
                  QuadraticOutput::costfuncdelta) {}

mat QuadraticOutput::costfunc(mat y, mat h) {
  const mat diff = y - h;
  const mat J = (diff % diff) / 2.0;
  return J;
}

mat QuadraticOutput::costfuncdelta(mat y, mat a, mat) {
  return a - y;
}

}
