#ifndef OUTPUT_EXCEPTION_H
#define OUTPUT_EXCEPTION_H

#include "NerdNet/except/nerdnet_exception.h"

namespace nerd {
namespace nn {
namespace except {

class OutputException : public NerdNetException {
 public:
  OutputException() : OutputException("OutputException") {}
  explicit OutputException(const char* msg)
      : NerdNetException(std::string("OutputException: ") + msg) {}
  explicit OutputException(const std::string& msg)
      : NerdNetException(std::string("OutputException: ") + msg) {}
};

} /* end of except namespace */
} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: OUTPUT_EXCEPTION_H */
