#include <cassert>
#include <chrono>
#include <random>
#include <iostream>
#include "tensor/tensor.h"

namespace nn {
namespace tensor {

using std::vector;
using std::mt19937;
using std::normal_distribution;

template <typename DType>
Tensor<DType>::Tensor() : data_(nullptr) {}

template <typename DType>
Tensor<DType>::Tensor(const DType val)
    : shape_(TensorShape()), data_(new DType{val}) {}

template <typename DType>
Tensor<DType>::~Tensor() {
  if (data_) delete[] data_;
}

template <typename DType>
Tensor<DType>::Tensor(const vector<int>& shape) : Tensor(TensorShape(shape)) {}

template <typename DType>
Tensor<DType>::Tensor(const TensorShape& shape) : shape_(shape) {
  // initialize with 0
  data_ = new DType[shape_.chunk(0)]{0};
}

template <typename DType>
Tensor<DType>::Tensor(const Tensor& tensor)
    : shape_(tensor.shape_), data_(nullptr) {
  data_ = new DType[shape_.chunk(0)]{0};
  for (int i = 0; i < shape_.chunk(0); ++i) data_[i] = tensor.data_[i];
}

template <typename DType>
Tensor<DType>::Tensor(Tensor&& tensor) : shape_(tensor.shape_), data_(nullptr) {
  Tensor<DType>::operator=(tensor);
}

template <typename DType>
Tensor<DType>& Tensor<DType>::operator=(const Tensor& tensor) {
  if (data_) {
    if (shape_.chunk(0) < tensor.shape_.chunk(0)) {
      delete[] data_;
      data_ = new DType[tensor.shape_.chunk(0)];
    }
    shape_ = tensor.shape_;
  } else {
    data_ = new DType[shape_.chunk(0)]{0};
  }
  for (int i = 0; i < shape_.chunk(0); ++i) data_[i] = tensor.data_[i];
  return *this;
}

template <typename DType>
Tensor<DType>&& Tensor<DType>::operator=(Tensor&& tensor) {
  data_ = tensor.data_;
  tensor.data_ = nullptr;
  return std::move(*this);
}

template <typename DType>
Tensor<DType> Tensor<DType>::Zeros(const TensorShape& shape) {
  return Tensor<DType>(shape);
}

template <typename DType>
Tensor<DType> Tensor<DType>::Zeros(const vector<int>& shape) {
  return Tensor<DType>::Zeros(TensorShape(shape));
}

template <typename DType>
Tensor<DType> Tensor<DType>::Ones(const TensorShape& shape) {
  Tensor<DType> output(shape);
  for (int i = 0; i < shape.chunk(0); ++i) {
    output.data_[i] = 1;
  }
  return output;
}

template <typename DType>
Tensor<DType> Tensor<DType>::Ones(const vector<int>& shape) {
  return Tensor<DType>::Ones(TensorShape(shape));
}

template <typename DType>
Tensor<DType> Tensor<DType>::Eyes(int size) {
  TensorShape shape({size, size});
  Tensor<DType> output(shape);
  for (int i = 0; i < size; ++i) {
    output.data_[i * size + i] = 1;
  }
  return output;
}

template <typename DType>
Tensor<DType> Tensor<DType>::Gaussian(const TensorShape& shape,
                                      DType mean, DType stddev) {
  mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
  normal_distribution<DType> dist(mean, stddev);
  Tensor<DType> output(shape);
  for (int i = 0; i < output.shape().chunk(0); ++i) {
    output.data_[i] = dist(gen);
  }
  return output;
}

template <typename DType>
Tensor<DType> Tensor<DType>::Gaussian(const vector<int>& shape,
                                      DType mean, DType stddev) {
  return Gaussian(TensorShape(shape), mean, stddev);
}

template <typename DType>
void Tensor<DType>::Reshape(const TensorShape& shape) {
  assert(shape.chunk(0) == shape_.chunk(0));
  shape_ = shape;
}

template <typename DType>
void Tensor<DType>::Reshape(const std::vector<int>& shape) {
  Reshape(TensorShape(shape));
}

template class Tensor<float>;
template class Tensor<double>;

} /* end of tensor namespace */
} /* end of nn namespace */
