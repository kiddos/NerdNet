#include "tensor/ops/reduce_sum.h"
#include "tensor/openmp_support.h"
#include <cassert>

namespace nn {
namespace tensor {

template <typename DType>
DType reduce_sum(const Tensor<DType>& t) {
  DType sum = 0;
  int chunk = t.shape().chunk(0);
  for (int i = 0 ; i < chunk ; ++i) {
    sum += t.data(i);
  }
  return sum;
}

template <typename DType>
Tensor<DType> reduce_sum(const Tensor<DType>& t, int axis) {
  assert(t.shape().size() == 2);

  int rows = t.shape().shape(0);
  int cols = t.shape().shape(1);
  if (axis == 0) {
    Tensor<DType> output(TensorShape({cols}));

    PARALLEL_FOR()
    for (int i = 0 ; i < cols ; ++i) {
      DType colsum = 0;
      for (int j = 0 ; j < rows ; ++j) {
        colsum += t.data(j * cols + i);
      }
      output[i] = colsum;
    }
    return output;
  } else {
    Tensor<DType> output(TensorShape({rows}));

    PARALLEL_FOR()
    for (int i = 0 ; i < rows ; ++i) {
      DType rowsum = 0;
      for (int j = 0 ; j < cols ; ++j) {
        rowsum += t.data(i * cols + j);
      }
      output[i] = rowsum;
    }
    return output;
  }
}

template float reduce_sum(const Tensor<float>& t);
template double reduce_sum(const Tensor<double>& t);
template Tensor<float> reduce_sum(const Tensor<float>& t, int axis);
template Tensor<double> reduce_sum(const Tensor<double>& t, int axis);

} /* end of tensor namespace */
} /* end of nn namespace */
