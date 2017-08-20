#include "NerdNet/layer/fc_layer.h"
#include "NerdNet/convert.h"
#include "NerdNet/except/variable_exception.h"
#include "NerdNet/layer/constant_initializer.h"
#include "NerdNet/layer/normal_initializer.h"

#include <boost/assert.hpp>
#include <cmath>

namespace nerd {
namespace nn {

FCLayer::FCLayer(BaseLayer* prev_layer, const VariableShape& var_shape)
    : VariableLayer(prev_layer, var_shape) {}

FCLayer::FCLayer(BaseLayer* prev_layer, const VariableShape& var_shape,
                 std::shared_ptr<VariableInitializer> weight_initializer,
                 std::shared_ptr<VariableInitializer> bias_initializer)
    : VariableLayer(prev_layer, var_shape, weight_initializer,
                    bias_initializer) {}

FCLayer::FCLayer(const FCLayer& layer)
    : FCLayer(layer.prev_layer_, layer.var_shape_, layer.weight_initializer_,
              layer.bias_initializer_) {
  FCLayer::operator=(layer);
}

FCLayer& FCLayer::operator=(const FCLayer& layer) {
  VariableLayer::operator=(layer);
  w_ = layer.w_;
  b_ = layer.b_;
  return *this;
}

Tensor<float> FCLayer::ForwardProp() {
  Tensor<float> input_tensor = prev_layer_->ForwardProp();
  Tensor2Matrix(input_tensor, input_);
  output_ = input_ * w_;
  output_.each_row() += b_;

  Tensor<float> output_tensor;
  Matrix2Tensor(output_, output_tensor);
  return output_tensor;
}

Tensor<float> FCLayer::BackProp(const Tensor<float>& delta_tensor) {
  arma::Mat<float> delta;
  Tensor2Matrix(delta_tensor, delta);

  arma::Mat<float> next_delta = delta * w_.t();
  w_grad_ = input_.t() * delta;
  b_grad_ = arma::sum(delta);

  Tensor<float> next_delta_tensor;
  Matrix2Tensor(next_delta, next_delta_tensor);
  return prev_layer_->BackProp(next_delta_tensor);
}

void FCLayer::Update(float learning_rate) {
  BOOST_ASSERT(learning_rate > 0);

  w_ -= learning_rate * w_grad_;
  b_ -= learning_rate * b_grad_;
}

bool FCLayer::Init() {
  // check
  if (var_shape_.dim() != 2) {
    throw except::VariableException("FCLayer incorrect variable dimension");
  }
  // initialize variables
  w_ = arma::Mat<float>(var_shape_[0], var_shape_[1]);
  b_ = arma::Row<float>(var_shape_[1]);

  int r = w_.n_rows;
  int c = w_.n_cols;
  if (weight_initializer_) {
    for (int i = 0; i < r; ++i) {
      for (int j = 0; j < c; ++j) {
        w_(i, j) = weight_initializer_->Next();
      }
    }
  } else {
    NormalInitializer init(0.0, std::sqrt(2.0 / r));
    for (int i = 0; i < r; ++i) {
      for (int j = 0; j < c; ++j) {
        w_(i, j) = init.Next();
      }
    }
  }

  if (bias_initializer_) {
    for (int i = 0; i < c; ++i) {
      b_(i) = bias_initializer_->Next();
    }
  } else {
    ConstantInitializer init(1e-2);
    for (int i = 0; i < c; ++i) {
      b_(i) = init.Next();
    }
  }
  return BaseLayer::Init();
}

} /* end of nn namespace */
} /* end of nerd namespace */
