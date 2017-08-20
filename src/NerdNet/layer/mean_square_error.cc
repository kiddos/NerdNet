#include "NerdNet/layer/mean_square_error.h"
#include "NerdNet/convert.h"
#include "NerdNet/except/input_exception.h"
#include "NerdNet/except/nullptr_exception.h"

namespace nerd {
namespace nn {

MeanSquareError::MeanSquareError(BaseLayer* input_layer,
                                 const VariableShape& var_shape)
    : FCLayer(input_layer, var_shape) {}

MeanSquareError::MeanSquareError(
    BaseLayer* input_layer, const VariableShape& var_shape,
    std::shared_ptr<VariableInitializer> weight_initializer,
    std::shared_ptr<VariableInitializer> bias_initializer)
    : FCLayer(input_layer, var_shape, weight_initializer, bias_initializer) {}

MeanSquareError::MeanSquareError(const MeanSquareError& layer)
    : FCLayer(layer) {}

MeanSquareError& MeanSquareError::operator=(const MeanSquareError& layer) {
  FCLayer::operator=(layer);
  return *this;
}

float MeanSquareError::ComputeCost() {
  Tensor<float> final_result_tensor = FCLayer::ForwardProp();
  Tensor2Matrix(final_result_tensor, final_result_);
  Tensor2Matrix(label_data_, label_);

  arma::Mat<float> diff = final_result_ - label_;
  arma::Mat<float> error = diff % diff;
  float cost = arma::accu(error) / 2.0f;
  return cost;
}

Tensor<float> MeanSquareError::ComputeDerivative() {
  return BackProp(Tensor<float>());
}

Tensor<float> MeanSquareError::BackProp(const Tensor<float>&) {
  arma::Mat<float> delta = final_result_ - label_;
  Tensor<float> next_delta_tensor;
  Matrix2Tensor(delta, next_delta_tensor);
  return FCLayer::BackProp(next_delta_tensor);
}

} /* end of nn namespace */
} /* end of nerd namespace */
