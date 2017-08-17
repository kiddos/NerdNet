#ifndef COST_FUNC_LAYER_H
#define COST_FUNC_LAYER_H

#include <armadillo>

#include "NerdNet/tensor.h"

namespace nerd {
namespace nn {

class CostFunction {
 public:
  CostFunction() = default;
  virtual ~CostFunction() {}

  virtual float ComputeCost() = 0;
  virtual Tensor<float> ComputeDerivative() = 0;

  void SetLabel(const Tensor<float>& label_data);
  void SetLabel(const arma::Mat<float>& label_data);
  void SetLabel(const arma::Cube<float>& label_data);
  void SetLabel(const arma::field<arma::Mat<float>>& label_data);
  void SetLabel(const arma::field<arma::Cube<float>>& label_data);

 protected:
  Tensor<float> label_data_;
};

} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: COST_FUNC_LAYER_H */
