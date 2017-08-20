#ifndef VARIABLE_LAYER_H
#define VARIABLE_LAYER_H

#include "NerdNet/layer/base_layer.h"
#include "NerdNet/layer/variable_initializer.h"

#include <memory>

namespace nerd {
namespace nn {

class VariableLayer : public BaseLayer {
 public:
  VariableLayer(BaseLayer* prev_layer, const VariableShape& var_shape);
  VariableLayer(BaseLayer* prev_layer, const VariableShape& var_shape,
                std::shared_ptr<VariableInitializer> weight_initializer,
                std::shared_ptr<VariableInitializer> bias_initializer);
  VariableLayer(const VariableLayer& layer);
  VariableLayer& operator=(const VariableLayer& layer);

  VariableShape shape() const { return var_shape_; }

 protected:
  std::shared_ptr<VariableInitializer> weight_initializer_, bias_initializer_;
  VariableShape var_shape_;
};

} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: VARIABLE_LAYER_H */
