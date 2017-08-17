#ifndef CONVERT_H
#define CONVERT_H

#include <armadillo>

#include "NerdNet/tensor.h"

namespace nerd {
namespace nn {

void Matrix2Tensor(const arma::Mat<float>& mat, Tensor<float>& tensor);
void Cube2Tensor(const arma::Cube<float>& cube, Tensor<float>& tensor);
void Matrices2Tensor(const arma::field<arma::Mat<float>>& matrices,
                     Tensor<float>& tensor);
void Cubes2Tensor(const arma::field<arma::Cube<float>>& cubes,
                  Tensor<float>& tensor);

void Tensor2Matrix(const Tensor<float>& tensor, arma::Mat<float>& mat);
void Tensor2Cube(const Tensor<float>& tensor, arma::Cube<float>& cube);
void Tensor2Matrices(const Tensor<float>& tensor,
                     arma::field<arma::Mat<float>>& matrices);
void Tensor2Cubes(const Tensor<float>& tensor,
                  arma::field<arma::Cube<float>>& cubes);

} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: CONVERT_H */
