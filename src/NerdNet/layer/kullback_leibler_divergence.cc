#include "NerdNet/layer/kullback_leibler_divergence.h"
#include "NerdNet/convert.h"

namespace nerd {
namespace nn {

KullbackLeiblerDivergence::KullbackLeiblerDivergence(
    BaseLayer* prev_layer, const VariableShape& var_shape)
    : FCLayer(prev_layer, var_shape) {}

KullbackLeiblerDivergence::KullbackLeiblerDivergence(
    BaseLayer* prev_layer, const VariableShape& var_shape,
    std::shared_ptr<VariableInitializer> weight_initializer,
    std::shared_ptr<VariableInitializer> bias_initializer)
    : FCLayer(prev_layer, var_shape, weight_initializer, bias_initializer) {}

KullbackLeiblerDivergence::KullbackLeiblerDivergence(
    const KullbackLeiblerDivergence& layer)
    : FCLayer(layer) {}

KullbackLeiblerDivergence& KullbackLeiblerDivergence::operator=(
    const KullbackLeiblerDivergence& layer) {
  FCLayer::operator=(layer);
  return *this;
}

Tensor<float> KullbackLeiblerDivergence::ForwardProp() {
  Tensor<float> fc_output_tensor = FCLayer::ForwardProp();
  Tensor2Matrix(fc_output_tensor, fc_output_);

  output_ = 1.0f / (1.0f + arma::exp(-fc_output_));
  Tensor<float> output_tensor;
  Matrix2Tensor(output_, output_tensor);
  return output_tensor;
}

float KullbackLeiblerDivergence::ComputeCost() {
  Tensor<float> final_result_tensor = ForwardProp();
  Tensor2Matrix(final_result_tensor, final_result_);
  Tensor2Matrix(label_data_, label_);

  arma::Mat<float> diff = final_result_ - label_;
  arma::Mat<float> error = label_ % arma::log(label_ / final_result_);
  float cost = arma::accu(error);
  return cost;
}

Tensor<float> KullbackLeiblerDivergence::ComputeDerivative() {
  return BackProp(Tensor<float>());
}

Tensor<float> KullbackLeiblerDivergence::BackProp(const Tensor<float>&) {
  arma::Mat<float> e = arma::exp(-fc_output_);
  arma::Mat<float> deriv = e / arma::pow(1.0f + e, 2);
  arma::Mat<float> delta = -label_ / final_result_ % deriv;
  Tensor<float> next_delta_tensor;
  Matrix2Tensor(delta, next_delta_tensor);
  return FCLayer::BackProp(next_delta_tensor);
}

} /* end of nn namespace */
} /* end of nerd namespace */
