#ifndef TANH_LAYER_H
#define TANH_LAYER_H

#include "NerdNet/layer/base_layer.h"

namespace nerd {
namespace nn {

class TanhLayer : public BaseLayer {
 public:
  TanhLayer(BaseLayer* prev_layer);

  Tensor<float> ForwardProp() override;
  Tensor<float> BackProp(const Tensor<float>& delta_tensor) override;
};

} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: TANH_LAYER_H */
