#include <armadillo>

#include "NerdNet/layer/sigmoid_layer.h"
#include "NerdNet/convert.h"

namespace nerd {
namespace nn {

SigmoidLayer::SigmoidLayer(BaseLayer* prev_layer) : BaseLayer(prev_layer) {}

Tensor<float> SigmoidLayer::ForwardProp() {
  Tensor<float> input_tensor = prev_layer_->ForwardProp();
  Tensor2Matrix(input_tensor, input_);

  output_ = 1.0f / (1.0f + arma::exp(-input_));
  Tensor<float> output_tensor;
  Matrix2Tensor(output_, output_tensor);
  return output_tensor;
}

Tensor<float> SigmoidLayer::BackProp(const Tensor<float>& delta_tensor) {
  arma::Mat<float> delta;
  Tensor2Matrix(delta_tensor, delta);
  arma::Mat<float> e = arma::exp(-input_);
  arma::Mat<float> deriv = e / arma::pow(1.0f + e, 2);
  arma::Mat<float> next_delta = deriv % delta;

  Tensor<float> next_delta_tensor;
  Matrix2Tensor(next_delta, next_delta_tensor);
  return next_delta_tensor;
}

} /* end of nn namespace */
} /* end of nerd namespace */
