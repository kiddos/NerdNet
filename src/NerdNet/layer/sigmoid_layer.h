#ifndef SIGMOID_LAYER_H
#define SIGMOID_LAYER_H

#include "NerdNet/layer/base_layer.h"

namespace nerd {
namespace nn {

class SigmoidLayer : public BaseLayer {
 public:
  SigmoidLayer(BaseLayer* prev_layer);

  Tensor<float> ForwardProp() override;
  Tensor<float> BackProp(const Tensor<float>& delta_tensor) override;
};

} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: SIGMOID_LAYER_H */
