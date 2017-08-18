#include <armadillo>

#include "NerdNet/convert.h"
#include "NerdNet/layer/leaky_relu_layer.h"

namespace nerd {
namespace nn {

LeakyReluLayer::LeakyReluLayer(BaseLayer* prev_layer)
    : BaseLayer(prev_layer), leak_(0.1f) {}

LeakyReluLayer::LeakyReluLayer(BaseLayer* prev_layer, float leak)
    : BaseLayer(prev_layer), leak_(leak) {}

Tensor<float> LeakyReluLayer::ForwardProp() {
  Tensor<float> input_tensor = prev_layer_->ForwardProp();
  Tensor2Matrix(input_tensor, input_);

  output_ = input_;
  output_.transform([this](float val) { return val < 0 ? leak_ * val : val; });
  Tensor<float> output_tensor;
  Matrix2Tensor(output_, output_tensor);
  return output_tensor;
}

Tensor<float> LeakyReluLayer::BackProp(const Tensor<float>& delta_tensor) {
  arma::Mat<float> delta;
  Tensor2Matrix(delta_tensor, delta);
  arma::Mat<float> deriv = input_;
  deriv.transform([this](float val) { return val < 0 ? leak_ : 1.0f; });
  arma::Mat<float> next_delta = deriv % delta;

  Tensor<float> next_delta_tensor;
  Matrix2Tensor(next_delta, next_delta_tensor);
  return prev_layer_->BackProp(next_delta_tensor);
}

} /* end of nn namespace */
} /* end of nerd namespace */
