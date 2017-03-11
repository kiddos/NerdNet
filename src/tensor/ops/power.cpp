#include "tensor/ops/power.h"
#include "tensor/openmp_support.h"
#include <cmath>
#include <cassert>

namespace nn {
namespace tensor {

template <typename DType>
Tensor<DType> operator^(const Tensor<DType>& t1, const Tensor<DType>& t2) {
  assert(t1.shape().size() == t2.shape().size());
  int shape_size = t1.shape().size();
  for (int i = 0; i < shape_size; ++i) {
    assert(t1.shape().shape(i) == t2.shape().shape(i));
  }

  Tensor<DType> output(t1.shape());
  int chunk = t1.shape().chunk(0);

  PARALLEL_FOR()
  for (int i = 0; i < chunk; ++i) {
    output[i] = std::pow(t1.data(i), t2.data(i));
  }
  return output;
}

template <typename DType>
Tensor<DType> operator^(const Tensor<DType>& t, const DType val) {
  Tensor<DType> output(t.shape());
  int chunk = t.shape().chunk(0);

  PARALLEL_FOR()
  for (int i = 0; i < chunk; ++i) {
    output[i] = std::pow(t.data(i), val);
  }
  return output;
}

template <typename DType>
Tensor<DType> power(const Tensor<DType>& t1, const Tensor<DType>& t2) {
  return t1 ^ t2;
}

template <typename DType>
Tensor<DType> power(const Tensor<DType>& t, const DType val) {
  return t ^ val;
}

template Tensor<float> operator^
    <float>(const Tensor<float>& t1, const Tensor<float>& t2);
template Tensor<double> operator^
    <double>(const Tensor<double>& t1, const Tensor<double>& t2);

template Tensor<float> operator^
    <float>(const Tensor<float>& t, const float val);
template Tensor<double> operator^
    <double>(const Tensor<double>& t, const double val);

template Tensor<float> power(const Tensor<float>& t1, const Tensor<float>& t2);
template Tensor<double> power(const Tensor<double>& t1,
                              const Tensor<double>& t2);

template Tensor<float> power(const Tensor<float>& t, const float val);
template Tensor<double> power(const Tensor<double>& t, const double val);

} /* end of tensor namespace */
} /* end of nn namespace */
