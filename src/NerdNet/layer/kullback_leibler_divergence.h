#ifndef KULLBACK_LEIBLER_DIVERGENCE_H
#define KULLBACK_LEIBLER_DIVERGENCE_H

#include "NerdNet/layer/cost_func_layer.h"
#include "NerdNet/layer/fc_layer.h"

namespace nerd {
namespace nn {

class KullbackLeiblerDivergence : public CostFunction, public FCLayer {
 public:
  KullbackLeiblerDivergence(BaseLayer* prev_layer,
                            const VariableShape& var_shape);
  KullbackLeiblerDivergence(
      BaseLayer* prev_layer, const VariableShape& var_shape,
      std::shared_ptr<VariableInitializer> weight_initializer,
      std::shared_ptr<VariableInitializer> bias_initializer);
  KullbackLeiblerDivergence(const KullbackLeiblerDivergence& layer);
  KullbackLeiblerDivergence& operator=(const KullbackLeiblerDivergence& layer);

  Tensor<float> ForwardProp() override;
  float ComputeCost() override;
  Tensor<float> ComputeDerivative() override;
  Tensor<float> BackProp(const Tensor<float>& delta_tensor) override;

 private:
  arma::Mat<float> fc_output_;
};

} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: KULLBACK_LEIBLER_DIVERGENCE_H */
