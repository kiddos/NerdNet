#include <cassert>
#include "tensor/tensor_shape.h"

namespace nn {
namespace tensor {

using std::vector;

TensorShape::TensorShape(const vector<int>& shape) : shape_(shape) {
  int shape_size = shape.size();
  assert(shape_size > 0);
  for (int i = 0 ; i < shape_size ; ++i) {
    if (i > 0) {
      assert(shape_[i] > 0);
    } else {
      assert(shape_[i] >= 0);
    }
  }
  // allocate chunk
  for (int i = 0 ; i < shape_size ; ++i) chunk_.push_back(0);
  chunk_[shape_size - 1] = shape[shape_size - 1];
  for (int i = shape_size - 2; i >= 0; --i) {
    chunk_[i] = shape[i] * chunk_[i + 1];
    assert(chunk_[i] >= 0);
  }
  if (chunk_[0] == 0) chunk_[0] = 1;
}

TensorShape::TensorShape(const TensorShape& tshape)
    : TensorShape(tshape.shape_) {}

TensorShape& TensorShape::operator=(const TensorShape& tshape) {
  TensorShape temp(tshape);
  shape_ = temp.shape_;
  chunk_ = temp.chunk_;
  return *this;
}

} /* end of tensor namespace */
} /* end of nn namespace */
