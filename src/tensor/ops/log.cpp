#include "tensor/ops/log.h"
#include "tensor/openmp_support.h"
#include <cmath>

namespace nn {
namespace tensor {

template <typename DType>
Tensor<DType> log(const Tensor<DType>& t) {
  Tensor<DType> output(t.shape());
  int chunk = t.shape().chunk(0);

  PARALLEL_FOR()
  for (int i = 0 ; i < chunk ; ++i) {
    output[i] = std::log(t.data(i));
  }
  return output;
}

template Tensor<float> log(const Tensor<float>& t);
template Tensor<double> log(const Tensor<double>& t);

} /* end of tensor namespace */
} /* end of nn namespace */
