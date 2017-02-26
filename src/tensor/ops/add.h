#ifndef TENSOR_ADD_H
#define TENSOR_ADD_H

#include "tensor/tensor.h"

namespace nn {
namespace tensor {

template <typename DType>
Tensor<DType> operator+(const Tensor<DType>& t1, const Tensor<DType>& t2);

template <typename DType>
Tensor<DType> add(const Tensor<DType>& t1, const Tensor<DType>& t2);

} /* end of tensor namespace */
} /* end of nn namespace */

#endif /* end of include guard: TENSOR_ADD_H */
