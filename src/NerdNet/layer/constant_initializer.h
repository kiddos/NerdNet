#ifndef CONSTANT_INITIALIZER_H
#define CONSTANT_INITIALIZER_H

#include "NerdNet/layer/variable_initializer.h"

namespace nerd {
namespace nn {

class ConstantInitializer : public VariableInitializer {
 public:
  ConstantInitializer(float value);

  float Next() override;

 private:
  float value_;
};

} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: CONSTANT_INITIALIZER_H */
