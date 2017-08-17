#ifndef INPUT_EXCEPTION_H
#define INPUT_EXCEPTION_H

#include "NerdNet/except/nerdnet_exception.h"

namespace nerd {
namespace nn {
namespace except {

class InputException : public NerdNetException {
 public:
  InputException() : InputException("InputException") {}
  explicit InputException(const char* msg)
      : NerdNetException(std::string("InputException: ") + msg) {}
  explicit InputException(const std::string& msg)
      : NerdNetException(std::string("InputException: ") + msg) {}
};

} /* end of except namespace */
} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: INPUT_EXCEPTION_H */
