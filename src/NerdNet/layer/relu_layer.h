#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include <armadillo>

#include "NerdNet/layer/base_layer.h"

namespace nerd {
namespace nn {

class ReluLayer : public BaseLayer {
 public:
  ReluLayer(BaseLayer* prev_layer);

  Tensor<float> ForwardProp() override;
  Tensor<float> BackProp(const Tensor<float>& delta_tensor) override;
};

} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: RELU_LAYER_H */
