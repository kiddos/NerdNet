#include <armadillo>

#include "NerdNet/convert.h"
#include "NerdNet/layer/smooth_relu_layer.h"

namespace nerd {
namespace nn {

SmoothReluLayer::SmoothReluLayer(BaseLayer* prev_layer)
    : BaseLayer(prev_layer) {}

Tensor<float> SmoothReluLayer::ForwardProp() {
  Tensor<float> input_tensor = prev_layer_->ForwardProp();
  Tensor2Matrix(input_tensor, input_);

  output_ = arma::log(1.0f + arma::exp(input_));
  Tensor<float> output_tensor;
  Matrix2Tensor(output_, output_tensor);
  return output_tensor;
}

Tensor<float> SmoothReluLayer::BackProp(const Tensor<float>& delta_tensor) {
  arma::Mat<float> delta;
  Tensor2Matrix(delta_tensor, delta);
  arma::Mat<float> e = arma::exp(input_);
  arma::Mat<float> deriv = e / (1.0f + e);
  arma::Mat<float> next_delta = deriv % delta;

  Tensor<float> next_delta_tensor;
  Matrix2Tensor(next_delta, next_delta_tensor);
  return prev_layer_->BackProp(next_delta_tensor);
}

} /* end of nn namespace */
} /* end of nerd namespace */
