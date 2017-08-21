#include "NerdNet/layer/sigmoid_cross_entropy.h"
#include "NerdNet/convert.h"

namespace nerd {
namespace nn {

SigmoidCrossEntropy::SigmoidCrossEntropy(BaseLayer* prev_layer,
                                         const VariableShape& var_shape)
    : FCLayer(prev_layer, var_shape) {}

SigmoidCrossEntropy::SigmoidCrossEntropy(
    BaseLayer* prev_layer, const VariableShape& var_shape,
    std::shared_ptr<VariableInitializer> weight_initializer,
    std::shared_ptr<VariableInitializer> bias_initializer)
    : FCLayer(prev_layer, var_shape, weight_initializer, bias_initializer) {}

SigmoidCrossEntropy::SigmoidCrossEntropy(const SigmoidCrossEntropy& layer)
    : FCLayer(layer) {}

SigmoidCrossEntropy& SigmoidCrossEntropy::operator=(
    const SigmoidCrossEntropy& layer) {
  FCLayer::operator=(layer);
  return *this;
}

Tensor<float> SigmoidCrossEntropy::ForwardProp() {
  Tensor<float> fc_output_tensor = FCLayer::ForwardProp();
  arma::Mat<float> fc_output;
  Tensor2Matrix(fc_output_tensor, fc_output);

  output_ = 1.0f / (1.0f + arma::exp(-fc_output));
  Tensor<float> output_tensor;
  Matrix2Tensor(output_, output_tensor);
  return output_tensor;
}

float SigmoidCrossEntropy::ComputeCost() {
  Tensor<float> final_result_tensor = ForwardProp();
  Tensor2Matrix(final_result_tensor, final_result_);
  Tensor2Matrix(label_data_, label_);

  arma::Mat<float> error = -label_ % arma::log(final_result_);
  float cost = arma::accu(error);
  return cost;
}

Tensor<float> SigmoidCrossEntropy::ComputeDerivative() {
  return BackProp(Tensor<float>());
}

Tensor<float> SigmoidCrossEntropy::BackProp(const Tensor<float>&) {
  arma::Mat<float> delta = label_ % (final_result_ - 1.0f);
  Tensor<float> next_delta_tensor;
  Matrix2Tensor(delta, next_delta_tensor);
  return FCLayer::BackProp(next_delta_tensor);
}

} /* end of nn namespace */
} /* end of nerd namespace */
