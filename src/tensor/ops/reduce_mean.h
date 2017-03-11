#ifndef TENSOR_REDUCE_MEAN_H
#define TENSOR_REDUCE_MEAN_H

#include "tensor/tensor.h"

namespace nn {
namespace tensor {

template <typename DType>
DType reduce_mean(const Tensor<DType>& t);

template <typename DType>
Tensor<DType> reduce_mean(const Tensor<DType>& t, int axis);

} /* end of tensor namespace */
} /* end of nn namespace */

#endif /* end of include guard: TENSOR_REDUCE_MEAN_H */
