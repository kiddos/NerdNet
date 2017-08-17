#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H

#include <armadillo>

#include "NerdNet/layer/base_layer.h"
#include "NerdNet/tensor.h"

namespace nerd {
namespace nn {

class InputLayer : public BaseLayer {
 public:
  InputLayer();
  InputLayer(const Tensor<float>& input_tensor);
  virtual ~InputLayer() {}

  void SetInput(const Tensor<float>& input_tensor);
  void SetInput(const arma::Mat<float>& input);
  void SetInput(const arma::Cube<float>& input);
  void SetInput(const arma::field<arma::Mat<float>>& input);
  void SetInput(const arma::field<arma::Cube<float>>& input);

  virtual Tensor<float> ForwardProp();
  virtual Tensor<float> BackProp(const Tensor<float>& delta_tensor);
  bool Init() override { return true; }

 private:
  Tensor<float> input_tensor_;
};

} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: INPUT_LAYER_H */
