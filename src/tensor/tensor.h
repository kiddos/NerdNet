#ifndef TENSOR_H
#define TENSOR_H

#include "tensor/tensor_shape.h"

namespace nn {
namespace tensor {

template <typename DType>
class Tensor {
 public:
  Tensor();
  ~Tensor();
  Tensor(const std::vector<int>& shape);
  Tensor(const TensorShape& shape);
  Tensor(const Tensor& tensor);
  Tensor(Tensor&& tensor);
  Tensor& operator=(const Tensor& tensor);
  Tensor&& operator=(Tensor&& tensor);

  static Tensor Zeros(const TensorShape& shape);
  static Tensor Zeros(const std::vector<int>& shape);
  static Tensor Ones(const TensorShape& shape);
  static Tensor Ones(const std::vector<int>& shape);
  static Tensor Eyes(int size);
  static Tensor Gaussian(const TensorShape& shape,
                         DType mean, DType stddev);
  static Tensor Gaussian(const std::vector<int>& shape,
                         DType mean, DType stddev);

  // reshape
  // total size must be equal
  void Reshape(const TensorShape& shape);
  void Reshape(const std::vector<int>& shape);

  TensorShape shape() const { return shape_; }
  DType data(int i) const { return data_[i]; }
  DType& operator[](int i) { return data_[i]; }
  template <typename T>
  T* ptr() { return reinterpret_cast<T*>(data_); }

 private:
  TensorShape shape_;
  DType* data_;
};

} /* end of tensor namespace */
} /* end of nn namespace */

#endif /* end of include guard: TENSOR_H */
