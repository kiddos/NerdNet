#ifndef ARCTAN_LAYER_H
#define ARCTAN_LAYER_H

#include "NerdNet/layer/base_layer.h"

namespace nerd {
namespace nn {

class ArctanLayer : public BaseLayer {
 public:
  ArctanLayer(BaseLayer* prev_layer);

  Tensor<float> ForwardProp() override;
  Tensor<float> BackProp(const Tensor<float>& delta_tensor) override;
};

} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: ARCTAN_LAYER_H */
