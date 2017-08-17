#ifndef SMOOTH_RELU_LAYER_H
#define SMOOTH_RELU_LAYER_H

#include "NerdNet/layer/base_layer.h"

namespace nerd {
namespace nn {

class SmoothReluLayer : public BaseLayer {
 public:
  SmoothReluLayer(BaseLayer* prev_layer);

  Tensor<float> ForwardProp() override;
  Tensor<float> BackProp(const Tensor<float>& delta_tensor) override;
};

} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: SMOOTH_RELU_LAYER_H */
