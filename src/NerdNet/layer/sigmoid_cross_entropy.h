#ifndef SIGMOID_CROSS_ENTROPY_H
#define SIGMOID_CROSS_ENTROPY_H

#include "NerdNet/layer/cost_func_layer.h"
#include "NerdNet/layer/fc_layer.h"

namespace nerd {
namespace nn {

class SigmoidCrossEntropy : public CostFunction, public FCLayer {
 public:
  SigmoidCrossEntropy(BaseLayer* prev_layer, const VariableShape& var_shape);
  SigmoidCrossEntropy(BaseLayer* prev_layer, const VariableShape& var_shape,
                      std::shared_ptr<VariableInitializer> weight_initializer,
                      std::shared_ptr<VariableInitializer> bias_initializer);
  SigmoidCrossEntropy(const SigmoidCrossEntropy& layer);
  SigmoidCrossEntropy& operator=(const SigmoidCrossEntropy& layer);

  Tensor<float> ForwardProp() override;
  float ComputeCost() override;
  Tensor<float> ComputeDerivative() override;
  Tensor<float> BackProp(const Tensor<float>& delta_tensor) override;
};

} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: SIGMOID_CROSS_ENTROPY_H */
