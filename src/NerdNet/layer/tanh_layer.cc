#include <armadillo>
#include <cmath>

#include "NerdNet/convert.h"
#include "NerdNet/layer/tanh_layer.h"

namespace nerd {
namespace nn {

TanhLayer::TanhLayer(BaseLayer* prev_layer) : BaseLayer(prev_layer) {}

Tensor<float> TanhLayer::ForwardProp() {
  Tensor<float> input_tensor = prev_layer_->ForwardProp();
  Tensor2Matrix(input_tensor, input_);

  output_ = input_;
  output_.transform([](float val) { return std::tanh(val); });
  Tensor<float> output_tensor;
  Matrix2Tensor(output_, output_tensor);
  return output_tensor;
}

Tensor<float> TanhLayer::BackProp(const Tensor<float>& delta_tensor) {
  arma::Mat<float> delta;
  Tensor2Matrix(delta_tensor, delta);
  arma::Mat<float> deriv = input_;
  deriv.transform(
      [](float val) { return 1.0f - std::pow(std::tanh(val), 2); });
  arma::Mat<float> next_delta = deriv % delta;

  Tensor<float> next_delta_tensor;
  Matrix2Tensor(next_delta, next_delta_tensor);
  return prev_layer_->BackProp(next_delta_tensor);
}

} /* end of nn namespace */
} /* end of nerd namespace */
