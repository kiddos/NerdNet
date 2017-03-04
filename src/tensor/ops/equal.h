#ifndef TENSOR_EQUAL_H
#define TENSOR_EQUAL_H

#include "tensor/tensor.h"

namespace nn {
namespace tensor {

template <typename DType>
Tensor<DType> operator==(const Tensor<DType>& t1, const Tensor<DType>& t2);

template <typename DType>
Tensor<DType> operator==(const Tensor<DType>& t, const DType val);

template <typename DType>
Tensor<DType> equal(const Tensor<DType>& t1, const Tensor<DType>& t2,
                    DType eps);

template <typename DType>
Tensor<DType> equal(const Tensor<DType>& t, const DType val, DType eps);

} /* end of tensor namespace */
} /* end of nn namespace */

#endif /* end of include guard: TENSOR_EQUAL_H */
