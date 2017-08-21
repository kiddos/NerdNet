#include "NerdNet/layer/input_layer.h"
#include "NerdNet/convert.h"
#include "NerdNet/except/input_exception.h"

namespace nerd {
namespace nn {

InputLayer::InputLayer() : BaseLayer(nullptr) {}

InputLayer::InputLayer(const Tensor<float>& input_tensor)
    : BaseLayer(nullptr), input_tensor_(input_tensor) {}

void InputLayer::SetInput(const Tensor<float>& input) {
  input_tensor_ = input;
}

void InputLayer::SetInput(const arma::Mat<float>& input) {
  Matrix2Tensor(input, input_tensor_);
}

void InputLayer::SetInput(const arma::Cube<float>& input) {
  Cube2Tensor(input, input_tensor_);
}

void InputLayer::SetInput(const arma::field<arma::Mat<float>>& input) {
  Matrices2Tensor(input, input_tensor_);
}

void InputLayer::SetInput(const arma::field<arma::Cube<float>>& input) {
  Cubes2Tensor(input, input_tensor_);
}

Tensor<float> InputLayer::ForwardProp() {
  if (input_tensor_.shape().size() == 0) {
    throw except::InputException("No given input");
  }
  return input_tensor_;
}

Tensor<float> InputLayer::BackProp(const Tensor<float>& delta_tensor) {
  return delta_tensor;
}

bool InputLayer::Init() { return true; }

} /* end of nn namespace */
} /* end of nerd namespace */
