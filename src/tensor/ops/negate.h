#ifndef TENSOR_NEGATE_H
#define TENSOR_NEGATE_H

#include "tensor/tensor.h"

namespace nn {
namespace tensor {

template <typename DType>
Tensor<DType> operator-(const Tensor<DType>& t);

template <typename DType>
Tensor<DType> neg(const Tensor<DType>& t);

} /* end of tensor namespace */
} /* end of nn namespace */

#endif /* end of include guard: TENSOR_NEGATE_H */
