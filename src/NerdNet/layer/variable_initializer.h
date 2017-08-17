#ifndef WEIGHT_INITIALIZER_H
#define WEIGHT_INITIALIZER_H

namespace nerd {
namespace nn {

class VariableInitializer {
 public:
  VariableInitializer() = default;
  virtual ~VariableInitializer() {}

  virtual float Next() = 0;
};

} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: WEIGHT_INITIALIZER_H */
