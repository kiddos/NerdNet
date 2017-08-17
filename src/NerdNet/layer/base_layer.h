#ifndef BASELAYER_H
#define BASELAYER_H

#include <armadillo>

#include "NerdNet/tensor.h"

namespace nerd {
namespace nn {

class BaseLayer {
 public:
  BaseLayer(BaseLayer* prev_layer);
  virtual ~BaseLayer() {}

  virtual Tensor<float> ForwardProp() = 0;
  virtual Tensor<float> BackProp(const Tensor<float>& delta_tensor) = 0;
  virtual void Update(float learning_rate);
  virtual bool Init();

 protected:
  BaseLayer* prev_layer_;
  arma::Mat<float> input_, output_;
};

} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: BASELAYER_H */
