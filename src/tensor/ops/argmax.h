#ifndef TENSOR_ARGMAX_H
#define TENSOR_ARGMAX_H

#include "tensor/tensor.h"

namespace nn {
namespace tensor {

template <typename DType>
Tensor<DType> argmax(const Tensor<DType>& t, int axis);

} /* end of tensor namespace */
} /* end of nn namespace */

#endif /* end of include guard: TENSOR_ARGMAX_H */
