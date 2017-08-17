#ifndef VARIABLE_EXCEPTION_H
#define VARIABLE_EXCEPTION_H

#include "NerdNet/except/nerdnet_exception.h"

namespace nerd {
namespace nn {
namespace except {

class VariableException : public NerdNetException {
 public:
  VariableException() : NerdNetException("VariableException") {}
  explicit VariableException(const char* msg)
      : NerdNetException(std::string("VariableException: ") + msg) {}
  explicit VariableException(const std::string& msg)
      : NerdNetException(std::string("VariableException: ") + msg) {}
};

} /* end of except namespace */
} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: VARIABLE_EXCEPTION_H */
