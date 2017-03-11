#include "tensor/ops/argmin.h"
#include "tensor/openmp_support.h"
#include <cassert>

namespace nn {
namespace tensor {

template <typename DType>
Tensor<DType> argmin(const Tensor<DType>& t, int axis) {
  assert(t.shape().size() == 2);

  int rows = t.shape().shape(0);
  int cols = t.shape().shape(1);
  if (axis == 0) {
    Tensor<DType> output(TensorShape({cols}));

    PARALLEL_FOR()
    for (int i = 0 ; i < cols ; ++i) {
      DType minval = t.data(i);
      for (int j = 1 ; j < rows ; ++j) {
        DType val = t.data(j * cols + i);
        if (val < minval) {
          minval = val;
        }
      }
      output[i] = minval;
    }
    return output;
  } else {
    Tensor<DType> output(TensorShape({rows}));

    PARALLEL_FOR()
    for (int i = 0 ; i < rows ; ++i) {
      DType minval = t.data(i * cols);
      for (int j = 1 ; j < cols ; ++j) {
        DType val = t.data(i * cols + j);
        if (val < minval) {
          minval = val;
        }
      }
      output[i] = minval;
    }
    return output;
  }
}

template Tensor<float> argmin(const Tensor<float>& t1, int axis);
template Tensor<double> argmin(const Tensor<double>& t1, int axis);

} /* end of tensor namespace */
} /* end of nn namespace */
