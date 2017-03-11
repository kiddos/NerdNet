#ifndef TENSOR_ARGMIN_H
#define TENSOR_ARGMIN_H

#include "tensor/tensor.h"

namespace nn {
namespace tensor {

template <typename DType>
Tensor<DType> argmin(const Tensor<DType>& t, int axis);

} /* end of tensor namespace */
} /* end of nn namespace */

#endif /* end of include guard: TENSOR_ARGMIN_H */
