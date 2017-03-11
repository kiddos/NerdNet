#ifndef TENSOR_SQRT_H
#define TENSOR_SQRT_H

#include "tensor/tensor.h"

namespace nn {
namespace tensor {

template <typename DType>
Tensor<DType> sqrt(const Tensor<DType>& t);

} /* end of tensor namespace */
} /* end of nn namespace */

#endif /* end of include guard: TENSOR_SQRT_H */
