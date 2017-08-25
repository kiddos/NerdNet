#include <armadillo>

#include "NerdNet/convert.h"
#include "NerdNet/except/input_exception.h"
#include "NerdNet/tensor.h"
#include "NerdNet/util/accuracy.h"

namespace nerd {
namespace nn {

template <>
double Accuracy(const arma::Mat<float>& prediction,
                const arma::Mat<float>& label) {
  if (prediction.n_rows != label.n_rows) {
    throw except::InputException("Incorrect prediction/label data size");
  }
  if (prediction.n_cols != label.n_cols) {
    throw except::InputException("Incorrect prediction/label data dimension");
  }

  int correct_count = 0;
  int data_size = prediction.n_rows;
  for (int i = 0; i < data_size; ++i) {
    arma::Row<float> p_row = prediction.row(i);
    arma::Row<float> l_row = label.row(i);
    arma::uvec p_index = arma::sort_index(p_row, "descend");
    arma::uvec l_index = arma::sort_index(l_row, "descend");
    if (p_index(0) == l_index(0)) ++correct_count;
  }
  return static_cast<double>(correct_count) / static_cast<double>(data_size);
}

template <>
double Accuracy(const Tensor<float>& prediction_tensor,
                const Tensor<float>& label_tensor) {
  arma::Mat<float> prediction, label;
  Tensor2Matrix(prediction_tensor, prediction);
  Tensor2Matrix(label_tensor, label);
  return Accuracy(prediction, label);
}

} /* end of nn namespace */
} /* end of nerd namespace */
