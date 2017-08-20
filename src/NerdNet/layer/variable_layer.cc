#include "NerdNet/layer/variable_layer.h"

namespace nerd {
namespace nn {

VariableLayer::VariableLayer(BaseLayer* prev_layer,
                             const VariableShape& var_shape)
    : BaseLayer(prev_layer), var_shape_(var_shape) {}

VariableLayer::VariableLayer(
    BaseLayer* prev_layer, const VariableShape& var_shape,
    std::shared_ptr<VariableInitializer> weight_initializer,
    std::shared_ptr<VariableInitializer> bias_initializer)
    : BaseLayer(prev_layer),
      weight_initializer_(weight_initializer),
      bias_initializer_(bias_initializer),
      var_shape_(var_shape) {}

VariableLayer::VariableLayer(const VariableLayer& layer)
    : VariableLayer(layer.prev_layer_, layer.var_shape_,
                    layer.weight_initializer_, layer.bias_initializer_) {}

VariableLayer& VariableLayer::operator=(const VariableLayer& layer) {
  prev_layer_ = layer.prev_layer_;
  var_shape_ = layer.var_shape_;
  weight_initializer_ = layer.weight_initializer_;
  bias_initializer_ = layer.bias_initializer_;
  return *this;
}

} /* end of nn namespace */
} /* end of nerd namespace */
