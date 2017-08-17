#ifndef MEAN_SQUARE_ERROR_H
#define MEAN_SQUARE_ERROR_H

#include "NerdNet/layer/cost_func_layer.h"
#include "NerdNet/layer/fc_layer.h"

namespace nerd {
namespace nn {

class MeanSquareError : public CostFunction, public FCLayer {
 public:
  MeanSquareError(BaseLayer* layer, const VariableShape& shape);
  MeanSquareError(BaseLayer* input_layer, const VariableShape& shape,
                  VariableInitializer* weight_initializer,
                  VariableInitializer* bias_initializer);
  MeanSquareError(const MeanSquareError& layer);
  MeanSquareError& operator=(const MeanSquareError& layer);

  float ComputeCost() override;
  Tensor<float> ComputeDerivative() override;
  Tensor<float> ForwardProp() override;
  Tensor<float> BackProp(const Tensor<float>& delta_tensor) override;

 protected:
  arma::Mat<float> final_result_, label_;
};

} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: MEAN_SQUARE_ERROR_H */
