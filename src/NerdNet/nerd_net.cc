#include "NerdNet/nerd_net.h"

namespace nerd {
namespace nn {

NerdNet::NerdNet() {
  has_variables_.push_back(false);
  layers_.push_back(std::unique_ptr<InputLayer>(new InputLayer));
}

bool NerdNet::Init() {
  return last()->Init();
}

} /* end of nn namespace */
} /* end of nerd namespace */
