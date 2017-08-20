#include "NerdNet/variable_shape.h"

namespace nerd {
namespace nn {

VariableShape::VariableShape(const std::vector<int>& shape) : shape_(shape) {}

VariableShape::VariableShape(const std::initializer_list<int>& shape)
    : shape_(shape) {}

VariableShape::VariableShape(const VariableShape& shape)
    : shape_(shape.shape_) {}

VariableShape& VariableShape::operator=(const VariableShape& shape) {
  shape_ = shape.shape_;
  return *this;
}

} /* end of nn namespace */
} /* end of nerd namespace */
