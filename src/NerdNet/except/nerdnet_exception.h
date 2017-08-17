#ifndef NERDNET_EXCEPTION_H
#define NERDNET_EXCEPTION_H

#include <exception>
#include <string>

namespace nerd {
namespace nn {
namespace except {

class NerdNetException : public std::exception {
 public:
  NerdNetException() : msg_("NerdNetException") {}
  explicit NerdNetException(const char* msg) : msg_(msg) {}
  explicit NerdNetException(const std::string& msg) : msg_(msg) {}
  virtual ~NerdNetException() {}

  virtual const char* what() const noexcept { return msg_.c_str(); }

 protected:
  std::string msg_;
};

} /* end of except namespace */
} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: NERDNET_EXCEPTION_H */
