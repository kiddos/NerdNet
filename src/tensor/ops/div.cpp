#include "tensor/ops/div.h"
#include <cassert>
#include <stdexcept>
#include <iostream>

namespace nn {
namespace tensor {

template <typename DType>
Tensor<DType> operator/(const Tensor<DType>& t1, const Tensor<DType>& t2) {
  int offset = t1.shape().size() - t2.shape().size();
  assert(offset >= 0);
  for (int i = 0; i < t2.shape().size(); ++i) {
    assert(t2.shape().shape(i) == t1.shape().shape(i + offset));
  }

  Tensor<DType> output(t1.shape());
  if (offset == 0) {
    int chunk = t1.shape().chunk(0);
    for (int i = 0; i < chunk; ++i) {
      try {
        output[i] = t1.data(i) / t2.data(i);
      } catch (std::logic_error e) {
        std::cerr << e.what() << std::endl;
      }
    }
  } else {
    int chunk = t1.shape().chunk(offset - 1);
    int mod = t1.shape().chunk(offset);
    for (int i = 0; i < chunk; ++i) {
      try {
        output[i] = t1.data(i) / t2.data(i % mod);
      } catch (std::logic_error e) {
        std::cerr << e.what() << std::endl;
      }
    }
  }
  return output;
}

template <typename DType>
Tensor<DType> operator/(const Tensor<DType>& t1, const DType val) {
  Tensor<DType> output(t1.shape());
  int chunk = t1.shape().chunk(0);
  for (int i = 0; i < chunk; ++i) {
    output[i] = t1.data(i) / val;
  }
  return output;
}

template <typename DType>
Tensor<DType> div(const Tensor<DType>& t1, const Tensor<DType>& t2) {
  return t1 / t2;
}

template <typename DType>
Tensor<DType> div(const Tensor<DType>& t1, const DType val) {
  return t1 / val;
}

template Tensor<float> operator/
    <float>(const Tensor<float>& t1, const Tensor<float>& t2);
template Tensor<float> operator/
    <float>(const Tensor<float>& t1, const float val);
template Tensor<double> operator/
    <double>(const Tensor<double>& t1, const Tensor<double>& t2);
template Tensor<double> operator/
    <double>(const Tensor<double>& t1, const double val);
template Tensor<float> div(const Tensor<float>& t1, const Tensor<float>& t2);
template Tensor<float> div(const Tensor<float>& t1, const float val);
template Tensor<double> div(const Tensor<double>& t1, const Tensor<double>& t2);
template Tensor<double> div(const Tensor<double>& t1, const double val);

} /* end of tensor namespace */
} /* end of nn namespace */

