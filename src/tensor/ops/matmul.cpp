#include "tensor/ops/matmul.h"
#include <cassert>
#include "tensor/openmp_support.h"
#include <iostream>

namespace nn {
namespace tensor {

template <typename DType>
Tensor<DType> operator%(const Tensor<DType>& t1, const Tensor<DType>& t2) {
  Tensor<DType> output;
  matmul(t1, t2, output);
  return output;
}

template <typename DType>
Tensor<DType> matmul(const Tensor<DType>& t1, const Tensor<DType>& t2,
                     Tensor<DType>& output) {
  assert((t1.shape().size() == 2) && (t2.shape().size() == 2));
  assert(t1.shape(1) == t2.shape(0));
  for (int i = 0; i < 2; ++i) {
    assert(t1.shape(i) > 0);
    assert(t2.shape(i) > 0);
  }

  int m = t1.shape(0);
  int k = t1.shape(1);
  int n = t2.shape(1);
  // reallocate memory if too small
  if (output.data() == nullptr || output.chunk(0) < m * n) {
    output = Tensor<DType>(TensorShape({m, n}));
  }

  for (int i = 0; i < m; ++i) {
    PARALLEL_FOR()
    for (int j = 0; j < n; ++j) {
      DType sum = 0;
      for (int l = 0; l < k; ++l) {
        sum += t1.data(i * k + l) * t2.data(l * n + j);
      }
      output[i * n + j] = sum;
    }
  }
  return output;
}

template Tensor<float> operator%(const Tensor<float>& t1,
                                 const Tensor<float>& t2);
template Tensor<double> operator%(const Tensor<double>& t1,
                                  const Tensor<double>& t2);
template Tensor<float> matmul(const Tensor<float>& t1, const Tensor<float>& t2,
                              Tensor<float>& output);
template Tensor<double> matmul(const Tensor<double>& t1,
                               const Tensor<double>& t2,
                               Tensor<double>& output);

} /* end of tensor namespace */
} /* end of nn namespace */
