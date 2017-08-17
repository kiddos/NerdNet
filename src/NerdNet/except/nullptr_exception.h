#ifndef NULLPTR_EXCEPTION_H
#define NULLPTR_EXCEPTION_H

#include "NerdNet/except/nerdnet_exception.h"

namespace nerd {
namespace nn {
namespace except {

class NullPtrException : public NerdNetException {
 public:
  NullPtrException() : NerdNetException("NullPtrException") {}
  explicit NullPtrException(const char* msg)
      : NerdNetException(std::string("NullPtrException: ") + msg) {}
  explicit NullPtrException(const std::string& msg)
      : NerdNetException(std::string("NullPtrException: ") + msg) {}
};

} /* end of except namespace */
} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: NULLPTR_EXCEPTION_H */
