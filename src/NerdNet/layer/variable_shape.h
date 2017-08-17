#ifndef VARIABLE_SHAPE_H
#define VARIABLE_SHAPE_H

#include <vector>

namespace nerd {
namespace nn {

class VariableShape {
 public:
  VariableShape() = default;
  explicit VariableShape(const std::vector<int>& shape);
  VariableShape(const std::initializer_list<int>& shape);
  VariableShape(const VariableShape& shape);
  VariableShape& operator=(const VariableShape& shape);

  int dim() const { return shape_.size(); }
  int operator[](int index) const { return shape_[index]; }

 private:
  std::vector<int> shape_;
};

} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: VARIABLE_SHAPE_H */
