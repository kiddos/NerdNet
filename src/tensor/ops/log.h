#ifndef TENSOR_LOG_H
#define TENSOR_LOG_H

#include "tensor/tensor.h"

namespace nn {
namespace tensor {

template <typename DType>
Tensor<DType> log(const Tensor<DType>& t);

} /* end of tensor namespace */
} /* end of nn namespace */

#endif /* end of include guard: TENSOR_LOG_H */
