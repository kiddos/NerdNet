#include "tensor/ops/sqrt.h"
#include "tensor/openmp_support.h"
#include <cmath>

namespace nn {
namespace tensor {

template <typename DType>
Tensor<DType> sqrt(const Tensor<DType>& t) {
  Tensor<DType> output(t.shape());
  int chunk = t.chunk(0);

  PARALLEL_FOR()
  for (int i = 0; i < chunk; ++i) {
    output[i] = std::sqrt(t.data(i));
  }
  return output;
}

template Tensor<float> sqrt<float>(const Tensor<float>& t);
template Tensor<double> sqrt<double>(const Tensor<double>& t);

} /* end of tensor namespace */
} /* end of nn namespace */
