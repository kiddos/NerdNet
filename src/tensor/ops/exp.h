#ifndef TENSOR_EXP_H
#define TENSOR_EXP_H

#include "tensor/tensor.h"

namespace nn {
namespace tensor {

template <typename DType>
Tensor<DType> exp(const Tensor<DType>& t);

} /* end of tensor namespace */
} /* end of nn namespace */

#endif /* end of include guard: TENSOR_EXP_H */
