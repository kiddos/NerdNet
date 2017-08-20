#include "NerdNet/layer/softmax_cross_entropy.h"
#include "NerdNet/convert.h"
#include "NerdNet/except/input_exception.h"
#include "NerdNet/except/nullptr_exception.h"

namespace nerd {
namespace nn {

SoftmaxCrossEntropy::SoftmaxCrossEntropy(BaseLayer* prev_layer,
                                         const VariableShape& var_shape)
    : FCLayer(prev_layer, var_shape) {}

SoftmaxCrossEntropy::SoftmaxCrossEntropy(
    BaseLayer* prev_layer, const VariableShape& var_shape,
    std::shared_ptr<VariableInitializer> weight_initializer,
    std::shared_ptr<VariableInitializer> bias_initializer)
    : FCLayer(prev_layer, var_shape, weight_initializer, bias_initializer) {}

SoftmaxCrossEntropy::SoftmaxCrossEntropy(const SoftmaxCrossEntropy& layer)
    : FCLayer(layer) {}

SoftmaxCrossEntropy& SoftmaxCrossEntropy::operator=(
    const SoftmaxCrossEntropy& layer) {
  FCLayer::operator=(layer);
  return *this;
}

Tensor<float> SoftmaxCrossEntropy::ForwardProp() {
  Tensor<float> fc_output_tensor = FCLayer::ForwardProp();
  arma::Mat<float> fc_output;
  Tensor2Matrix(fc_output_tensor, fc_output);

  arma::Mat<float> e = arma::exp(fc_output.each_col() - arma::max(fc_output, 1));
  output_ = e.each_col() / arma::sum(e, 1);
  Tensor<float> output_tensor;
  Matrix2Tensor(output_, output_tensor);
  return output_tensor;
}

float SoftmaxCrossEntropy::ComputeCost() {
  Tensor<float> final_result_tensor = ForwardProp();
  Tensor2Matrix(final_result_tensor, final_result_);
  Tensor2Matrix(label_data_, label_);

  arma::Mat<float> error = - label_ % arma::log(final_result_);
  float cost = arma::accu(error);
  return cost;
}

Tensor<float> SoftmaxCrossEntropy::ComputeDerivative() {
  return BackProp(Tensor<float>());
}

Tensor<float> SoftmaxCrossEntropy::BackProp(
    const Tensor<float>&) {
  arma::Mat<float> delta = final_result_ - label_;
  Tensor<float> next_delta_tensor;
  Matrix2Tensor(delta, next_delta_tensor);
  return FCLayer::BackProp(next_delta_tensor);
}

} /* end of nn namespace */
} /* end of nerd namespace */
