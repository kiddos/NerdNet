#ifndef TENSOR_SHAPE_H
#define TENSOR_SHAPE_H

#include <vector>

namespace nn {
namespace tensor {

class TensorShape {
 public:
  TensorShape() = default;
  TensorShape(const std::vector<int>& shape);
  TensorShape(const TensorShape& tshape);
  TensorShape& operator=(const TensorShape& tshape);
  int shape(int i) const { return shape_[i]; }
  int chunk(int i) const { return chunk_[i]; }
  int size() const { return shape_.size(); }

 private:
  std::vector<int> shape_, chunk_;
};

} /* end of tensor namespace */
} /* end of nn namespace */

#endif /* end of include guard: TENSOR_SHAPE_H */
