#include "NerdNet/layer/cost_func_layer.h"
#include "NerdNet/convert.h"

namespace nerd {
namespace nn {

void CostFunction::SetLabel(const Tensor<float>& label_data) {
  label_data_ = label_data;
}

void CostFunction::SetLabel(const arma::Mat<float>& label_data) {
  Matrix2Tensor(label_data, label_data_);
}

void CostFunction::SetLabel(const arma::Cube<float>& label_data) {
  Cube2Tensor(label_data, label_data_);
}

void CostFunction::SetLabel(const arma::field<arma::Mat<float>>& label_data) {
  Matrices2Tensor(label_data, label_data_);
}

void CostFunction::SetLabel(const arma::field<arma::Cube<float>>& label_data) {
  Cubes2Tensor(label_data, label_data_);
}

} /* end of nn namespace */
} /* end of nerd namespace */
