#include <armadillo>

#include "NerdNet/layer/relu_layer.h"
#include "NerdNet/convert.h"

namespace nerd {
namespace nn {

ReluLayer::ReluLayer(BaseLayer* prev_layer) : BaseLayer(prev_layer) {}

Tensor<float> ReluLayer::ForwardProp() {
  Tensor<float> input_tensor = prev_layer_->ForwardProp();
  Tensor2Matrix(input_tensor, input_);

  output_ = input_;
  output_.transform([](float val) { return val < 0 ? 0 : val; });
  Tensor<float> output_tensor;
  Matrix2Tensor(output_, output_tensor);
  return output_tensor;
}

Tensor<float> ReluLayer::BackProp(const Tensor<float>& delta_tensor) {
  arma::Mat<float> delta;
  Tensor2Matrix(delta_tensor, delta);
  arma::Mat<float> deriv = input_;
  deriv.transform([](float val) { return val < 0 ? 0 : 1.0f; });
  arma::Mat<float> next_delta = deriv % delta;

  Tensor<float> next_delta_tensor;
  Matrix2Tensor(next_delta, next_delta_tensor);
  return next_delta_tensor;
}

} /* end of nn namespace */
} /* end of nerd namespace */
