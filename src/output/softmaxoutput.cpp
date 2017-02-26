#include "softmaxoutput.h"

namespace nn {

SoftmaxOutput::SoftmaxOutput() {}

SoftmaxOutput::SoftmaxOutput(const SoftmaxOutput& output)
    : OutputLayer(output) {}

SoftmaxOutput::SoftmaxOutput(const int pnnodes, const int outputnodes,
                             const double lrate, const double lambda)
    : OutputLayer(pnnodes, outputnodes, lrate, lambda, identity,
                  SoftmaxOutput::costfunc,
                  SoftmaxOutput::costfuncdelta) {}

SoftmaxOutput::SoftmaxOutput(const int pnnodes, const int outputnodes,
                             const double lrate, const double stddev,
                             const double lambda)
    : OutputLayer(pnnodes, outputnodes, lrate, stddev, lambda, identity,
                  SoftmaxOutput::costfunc,
                  SoftmaxOutput::costfuncdelta) {}

SoftmaxOutput::SoftmaxOutput(LayerParam param)
    : OutputLayer(param, SoftmaxOutput::costfunc,
                  SoftmaxOutput::costfuncdelta) {}

mat SoftmaxOutput::costfunc(mat y, mat h) {
  const mat expo = exponential(h);
  const mat sumexpo = repeat(rowsum(expo), 1, y.n_cols);
  const mat P = expo % (1.0 / sumexpo);
  const mat J = - (y % arma::log(P)) / y.n_rows;
  return J;
}

mat SoftmaxOutput::costfuncdelta(mat y, mat a, mat) {
  const mat expo = exponential(a);
  const mat sumexpo = repeat(rowsum(expo), 1, y.n_cols);
  const mat P = expo % (1.0 / sumexpo);
  const mat delta = P - y;
  return delta / y.n_rows;
}

}
