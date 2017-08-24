#include "NerdNet/tensor.h"

#include <cstring>

namespace nerd {
namespace nn {

template <typename T>
Tensor<T>::Tensor(T value) : push_index_(0), shape_({0}), data_({value}) {}

template <typename T>
Tensor<T>::Tensor(const std::vector<int>& shape)
    : push_index_(0), shape_(shape) {
  int shape_size = shape.size();
  if (shape_size > 0) {
    int data_size = shape[0];
    for (int i = 0; i < shape_size; ++i) {
      data_size *= shape[i];
    }
    data_ = std::vector<T>(data_size);
  };
}

template <typename T>
Tensor<T>::Tensor(const VariableShape& shape) : Tensor(shape.shape()) {}

template <typename T>
Tensor<T>::Tensor(std::initializer_list<int> shape)
    : Tensor(std::vector<int>(shape)) {}

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

  int data_size = tensor.data_.size();
  data_ = std::vector<T>(data_size);
  std::memcpy(&data_[0], tensor.data(), data_size * sizeof(T));

  return *this;
}

template <typename T>
T Tensor<T>::operator<<(T value) {
  data_[push_index_++] = value;
  int data_size = data_.size();
  if (push_index_ >= data_size) push_index_ = 0;
  return value;
}

template class Tensor<float>;
template class Tensor<double>;

} /* end of nn namespace */
} /* end of nerd namespace */
