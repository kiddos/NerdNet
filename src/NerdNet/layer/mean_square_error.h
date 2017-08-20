#ifndef MEAN_SQUARE_ERROR_H
#define MEAN_SQUARE_ERROR_H

#include "NerdNet/layer/cost_func_layer.h"
#include "NerdNet/layer/fc_layer.h"

namespace nerd {
namespace nn {

class MeanSquareError : public CostFunction, public FCLayer {
 public:
  MeanSquareError(BaseLayer* prev_layer, const VariableShape& shape);
  MeanSquareError(BaseLayer* prev_layer, const VariableShape& shape,
                  std::shared_ptr<VariableInitializer> weight_initializer,
                  std::shared_ptr<VariableInitializer> bias_initializer);
  MeanSquareError(const MeanSquareError& layer);
  MeanSquareError& operator=(const MeanSquareError& layer);

  float ComputeCost() override;
  Tensor<float> ComputeDerivative() override;
  Tensor<float> BackProp(const Tensor<float>& delta_tensor) override;
};

} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: MEAN_SQUARE_ERROR_H */
