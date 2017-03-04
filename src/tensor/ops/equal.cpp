#include "tensor/ops/equal.h"
#include <cassert>
#include <cmath>

namespace nn {
namespace tensor {

template <typename DType>
Tensor<DType> operator==(const Tensor<DType>& t1, const Tensor<DType>& t2) {
  assert(t1.shape().size() == t2.shape().size());
  int shape_size = t1.shape().size();
  for (int i = 0; i < shape_size; ++i) {
    assert(t1.shape().shape(i) == t1.shape().shape(i));
  }

  Tensor<DType> output(t1.shape());
  int chunk = t1.shape().chunk(0);
  for (int i = 0; i < chunk; ++i) {
    if (t1.data(i) == t2.data(i)) {
      output[i] = 1;
    }
  }
  return output;
}

template <typename DType>
Tensor<DType> operator==(const Tensor<DType>& t, const DType val) {
  Tensor<DType> output(t.shape());
  int chunk = t.shape().chunk(0);
  for (int i = 0; i < chunk; ++i) {
    if (t.data(i) == val) {
      output[i] = 1;
    }
  }
  return output;
}

template <typename DType>
Tensor<DType> equal(const Tensor<DType>& t1, const Tensor<DType>& t2,
                    DType eps) {
  assert(eps >= 0);
  assert(t1.shape().size() == t2.shape().size());
  int shape_size = t1.shape().size();
  for (int i = 0; i < shape_size; ++i) {
    assert(t1.shape().shape(i) == t1.shape().shape(i));
  }

  Tensor<DType> output(t1.shape());
  int chunk = t1.shape().chunk(0);
  for (int i = 0; i < chunk; ++i) {
    if (std::abs(t1.data(i) - t2.data(i)) < eps) {
      output[i] = 1;
    }
  }
  return output;
}

template <typename DType>
Tensor<DType> equal(const Tensor<DType>& t, const DType val, DType eps) {
  assert(eps >= 0);
  Tensor<DType> output(t.shape());
  int chunk = t.shape().chunk(0);
  for (int i = 0; i < chunk; ++i) {
    if (std::abs(t.data(i) - val) < eps) {
      output[i] = 1;
    }
  }
  return output;
}

template Tensor<float> operator==(const Tensor<float>& t1,
                                  const Tensor<float>& t2);
template Tensor<double> operator==(const Tensor<double>& t1,
                                   const Tensor<double>& t2);
template Tensor<float> operator==(const Tensor<float>& t, const float val);
template Tensor<double> operator==(const Tensor<double>& t, const double val);
template Tensor<float> equal(const Tensor<float>& t1, const Tensor<float>& t2,
                             float eps);
template Tensor<double> equal(const Tensor<double>& t1,
                              const Tensor<double>& t2, double eps);
template Tensor<float> equal(const Tensor<float>& t, const float val,
                             float eps);
template Tensor<double> equal(const Tensor<double>& t, const double val,
                              double eps);

} /* end of tensor namespace */
} /* end of nn namespace */
