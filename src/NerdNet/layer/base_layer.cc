#include "NerdNet/layer/base_layer.h"
#include "NerdNet/except/nullptr_exception.h"

namespace nerd {
namespace nn {

BaseLayer::BaseLayer(BaseLayer* prev_layer) : prev_layer_(prev_layer) {}

void BaseLayer::Update(float) {}

bool BaseLayer::Init() {
  if (!prev_layer_) {
    throw except::NullPtrException("Previous layer null pointer");
  }
  // recursively initialize
  return prev_layer_->Init();
}

} /* end of nn namespace */
} /* end of nerd namespace */
