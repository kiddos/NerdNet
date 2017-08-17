#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>

namespace nerd {
namespace nn {

template <typename T>
class Tensor {
 public:
  Tensor() = default;
  explicit Tensor(T value);
  Tensor(int data_size, const std::vector<int>& shape);
  Tensor(const T* data, int data_size, const std::vector<int>& shape);
  Tensor(const Tensor<T>& tensor);
  Tensor<T>& operator=(const Tensor<T>& tensor);

  int operator[](int index) const { return shape_[index]; }
  std::vector<int> shape() const { return shape_; }
  const T* data() const { return &data_[0]; }
  T* mutable_data() { return &data_[0]; }

 private:
  std::vector<int> shape_;
  std::vector<T> data_;
};

template <typename T>
std::ostream& operator<<(std::ostream& out, const Tensor<T>& tensor) {
  int shape_size = tensor.shape().size();
  out << "Tensor[shape:(";
  for (int i = 0; i < shape_size; ++i) {
    out << tensor.shape()[i];
    if (i < shape_size - 1) out << ",";
  }
  out << ")]";
  return out;
}

} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: TENSOR_H */
