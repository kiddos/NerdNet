#ifndef LEAKY_RELU_H
#define LEAKY_RELU_H

#include "NerdNet/layer/base_layer.h"

namespace nerd {
namespace nn {

class LeakyReluLayer : public BaseLayer {
 public:
  LeakyReluLayer(BaseLayer* prev_layer);
  LeakyReluLayer(BaseLayer* prev_layer, float leak);

  Tensor<float> ForwardProp() override;
  Tensor<float> BackProp(const Tensor<float>& delta_tensor) override;

 private:
  float leak_;
};

} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: LEAKY_RELU_H */
