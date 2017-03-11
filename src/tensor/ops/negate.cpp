#include "tensor/ops/negate.h"
#include "tensor/openmp_support.h"

namespace nn {
namespace tensor {

template <typename DType>
Tensor<DType> operator-(const Tensor<DType>& t) {
  Tensor<DType> output(t.shape());
  int chunk = t.shape().chunk(0);

  PARALLEL_FOR()
  for (int i = 0 ; i < chunk ; ++i) {
    output[i] = - t.data(i);
  }
  return output;
}

template <typename DType>
Tensor<DType> neg(const Tensor<DType>& t) {
  return -t;
}

template Tensor<float> operator-(const Tensor<float>& t);
template Tensor<double> operator-(const Tensor<double>& t);
template Tensor<float> neg(const Tensor<float>& t);
template Tensor<double> neg(const Tensor<double>& t);

} /* end of tensor namespace */
} /* end of nn namespace */
