#ifndef TRAINER_EXCEPTION_H
#define TRAINER_EXCEPTION_H

#include "NerdNet/except/nerdnet_exception.h"

namespace nerd {
namespace nn {
namespace except {

class TrainerException : public NerdNetException {
 public:
  TrainerException() : NerdNetException("TrainerException") {}
  explicit TrainerException(const char* msg)
      : NerdNetException(std::string("TrainerException: ") + msg) {}
  explicit TrainerException(const std::string& msg)
      : NerdNetException(std::string("TrainerException: ") + msg) {}
};

} /* end of except namespace */
} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: TRAINER_EXCEPTION_H */
