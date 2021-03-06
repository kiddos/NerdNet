#ifndef TENSOR_SUB_H
#define TENSOR_SUB_H

#include "tensor/tensor.h"

namespace nn {
namespace tensor {

template <typename DType>
Tensor<DType> operator-(const Tensor<DType>& t1, const Tensor<DType>& t2);

template <typename DType>
Tensor<DType> operator-(const Tensor<DType>& t1, const DType val);

template <typename DType>
Tensor<DType> sub(const Tensor<DType>& t1, const Tensor<DType>& t2);

template <typename DType>
Tensor<DType> sub(const Tensor<DType>& t1, const DType val);

} /* end of tensor namespace */
} /* end of nn namespace */

#endif /* end of include guard: TENSOR_SUB_H */
