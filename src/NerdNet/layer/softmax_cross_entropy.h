#ifndef SOFTMAX_CROSS_ENTROPY_H
#define SOFTMAX_CROSS_ENTROPY_H

#include "NerdNet/layer/cost_func_layer.h"
#include "NerdNet/layer/fc_layer.h"

namespace nerd {
namespace nn {

class SoftmaxCrossEntropy : public CostFunction, public FCLayer {
 public:
  SoftmaxCrossEntropy(BaseLayer* prev_layer, const VariableShape& shape);
  SoftmaxCrossEntropy(BaseLayer* prev_layer, const VariableShape& shape,
                      std::shared_ptr<VariableInitializer> weight_initializer,
                      std::shared_ptr<VariableInitializer> bias_initializer);
  SoftmaxCrossEntropy(const SoftmaxCrossEntropy& layer);
  SoftmaxCrossEntropy& operator=(const SoftmaxCrossEntropy& layer);

  Tensor<float> ForwardProp() override;
  float ComputeCost() override;
  Tensor<float> ComputeDerivative() override;
  Tensor<float> BackProp(const Tensor<float>& delta_tensor) override;
};

} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: SOFTMAX_CROSS_ENTROPY_H */
