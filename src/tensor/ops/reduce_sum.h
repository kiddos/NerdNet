#ifndef TENSOR_REDUCE_SUM_H
#define TENSOR_REDUCE_SUM_H

#include "tensor/tensor.h"

namespace nn {
namespace tensor {

template <typename DType>
DType reduce_sum(const Tensor<DType>& t);

template <typename DType>
Tensor<DType> reduce_sum(const Tensor<DType>& t, int axis);

} /* end of tensor namespace */
} /* end of nn namespace */

#endif /* end of include guard: TENSOR_REDUCE_SUM_H */
