#include "NerdNet/tensor.h"

#include <cstring>

namespace nerd {
namespace nn {

template <typename T>
Tensor<T>::Tensor(T value) : shape_({0}), data_({value}) {}

template <typename T>
Tensor<T>::Tensor(int data_size, const std::vector<int>& shape)
    : shape_(shape), data_(data_size) {}

template <typename T>
Tensor<T>::Tensor(const T* data, int data_size, const std::vector<int>& shape)
    : shape_(shape), data_(data_size) {
  std::memcpy(&data_[0], data, data_size * sizeof(T));
}

template <typename T>
Tensor<T>::Tensor(const Tensor<T>& tensor) {
  Tensor::operator=(tensor);
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>& tensor) {
  shape_ = tensor.shape();

  int shape_size = tensor.shape().size();
  int data_size = tensor[0];
  for (int i = 1; i < shape_size; ++i) {
    data_size *= tensor.shape()[i];
  }

  data_ = std::vector<T>(data_size);
  std::memcpy(&data_[0], tensor.data(), data_size * sizeof(T));

  return *this;
}

template class Tensor<float>;
template class Tensor<double>;

} /* end of nn namespace */
} /* end of nerd namespace */
