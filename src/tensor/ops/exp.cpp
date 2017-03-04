#include "tensor/ops/exp.h"
#include <cmath>

namespace nn {
namespace tensor {

template <typename DType>
Tensor<DType> exp(const Tensor<DType>& t) {
  Tensor<DType> output(t.shape());
  int chunk = t.shape().chunk(0);
  for (int i = 0 ; i < chunk ; ++i) {
    output[i] = std::exp(t.data(i));
  }
  return output;
}

template Tensor<float> exp(const Tensor<float>& t);
template Tensor<double> exp(const Tensor<double>& t);

} /* end of tensor namespace */
} /* end of nn namespace */
