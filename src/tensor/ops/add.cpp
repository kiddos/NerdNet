#include <cassert>
#include "tensor/ops/add.h"

namespace nn {
namespace tensor {

template <typename DType>
Tensor<DType> operator+(const Tensor<DType>& t1, const Tensor<DType>& t2) {
  int offset = t1.shape().size() - t2.shape().size();
  assert(offset >= 0);
  for (int i = 0 ; i < t2.shape().size() ; ++i) {
    assert(t2.shape().shape(i) == t1.shape().shape(i + offset));
  }

  Tensor<DType> output(t1.shape());
  if (offset == 0) {
    int chunk = t1.shape().chunk(0);
    for (int i = 0 ; i < chunk ; ++i) {
      output[i] = t1.data(i) + t2.data(i);
    }
  } else {
    int chunk = t1.shape().chunk(offset);
    int shape = t1.shape().shape(offset-1);
    for (int i = 0 ; i < shape ; ++i) {
      for (int j = 0 ; j < chunk ; ++j) {
        output[i * chunk + j] = t1.data(i * chunk + j) + t2.data(j);
      }
    }
  }
  return output;
}

template <typename DType>
Tensor<DType> add(const Tensor<DType>& t1, const Tensor<DType>& t2) {
  return t1 + t2;
}

template
Tensor<float> operator+<float>(const Tensor<float>& t1,
                               const Tensor<float>& t2);
template
Tensor<double> operator+<double>(const Tensor<double>& t1,
                                 const Tensor<double>& t2);
template
Tensor<float> add(const Tensor<float>& t1, const Tensor<float>& t2);
template
Tensor<double> add(const Tensor<double>& t1, const Tensor<double>& t2);

} /* end of tensor namespace */
} /* end of nn namespace */