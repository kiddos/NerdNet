#ifndef FC_LAYER_H
#define FC_LAYER_H

#include "NerdNet/layer/variable_layer.h"

namespace nerd {
namespace nn {

class FCLayer : public VariableLayer {
 public:
  FCLayer(BaseLayer* prev_layer, const VariableShape& shape);
  FCLayer(BaseLayer* prev_layer, const VariableShape& shape,
          std::shared_ptr<VariableInitializer> weight_initializer,
          std::shared_ptr<VariableInitializer> bias_initializer);
  FCLayer(const FCLayer& layer);
  FCLayer& operator=(const FCLayer& layer);

  int input_size() const { return var_shape_[0]; }
  int output_size() const { return var_shape_[1]; }
  arma::Mat<float> input() const { return input_; }
  arma::Mat<float> output() const { return output_; }
  arma::Mat<float> weight() const { return w_; }
  arma::Mat<float> weight_gradient() const { return w_grad_; }
  arma::Row<float> bias() const { return b_; }
  arma::Row<float> bias_gradient() const { return b_grad_; }
  void set_weight(const arma::Mat<float>& w) { w_ = w; }
  void set_bias(const arma::Row<float>& b) { b_ = b; }

  Tensor<float> ForwardProp() override;
  Tensor<float> BackProp(const Tensor<float>& delta_tensor) override;
  void Update(float learning_rate) override;
  bool Init() override;

 protected:
  arma::Mat<float> w_, w_grad_;
  arma::Row<float> b_, b_grad_;
};

} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: FC_LAYER_H */
