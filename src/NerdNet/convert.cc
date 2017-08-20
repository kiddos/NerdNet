#include "NerdNet/convert.h"
#include <boost/assert.hpp>

namespace nerd {
namespace nn {

void Matrix2Tensor(const arma::Mat<float>& mat, Tensor<float>& tensor) {
  int r = mat.n_rows;
  int c = mat.n_cols;
  tensor = Tensor<float>(mat.memptr(), r * c, {r, c});
}

void Cube2Tensor(const arma::Cube<float>& cube, Tensor<float>& tensor) {
  int r = cube.n_rows;
  int c = cube.n_cols;
  int s = cube.n_slices;
  tensor = Tensor<float>(cube.memptr(), r * c * s, {r, c, s});
}

void Matrices2Tensor(const arma::field<arma::Mat<float>>& matrices,
                     Tensor<float>& tensor) {
  if (matrices.n_elem > 0) {
    arma::Mat<float> m = matrices(0);
    int r = m.n_rows;
    int c = m.n_cols;
    int e = matrices.n_elem;
    tensor = Tensor<float>({e, r, c});

    int mat_size = r * c;
    float* ptr = tensor.mutable_data();
    for (int i = 0; i < e; ++i) {
      arma::Mat<float> m = matrices(i);
      std::memcpy(ptr, m.memptr(), mat_size * sizeof(float));
      ptr += mat_size;
    }
  } else {
    tensor = Tensor<float>();
  }
}

void Cubes2Tensor(const arma::field<arma::Cube<float>>& cubes,
                  Tensor<float>& tensor) {
  if (cubes.n_elem > 0) {
    arma::Cube<float> cu = cubes(0);
    int r = cu.n_rows;
    int c = cu.n_cols;
    int s = cu.n_slices;
    int e = cubes.n_elem;
    tensor = Tensor<float>({e, r, c, s});

    int cube_size = r * c * s;
    float* ptr = tensor.mutable_data();
    for (int i = 0; i < e; ++i) {
      arma::Cube<float> cu = cubes(i);
      std::memcpy(ptr, cu.memptr(), cube_size * sizeof(float));
      ptr += cube_size;
    }
  } else {
    tensor = Tensor<float>();
  }
}

void Tensor2Matrix(const Tensor<float>& tensor, arma::Mat<float>& mat) {
  BOOST_ASSERT(tensor.shape().size() == 2);
  mat = arma::Mat<float>(tensor.data(), tensor[0], tensor[1]);
}

void Tensor2Cube(const Tensor<float>& tensor, arma::Cube<float>& cube) {
  BOOST_ASSERT(tensor.shape().size() == 3);
  cube = arma::Cube<float>(tensor.data(), tensor[0], tensor[1], tensor[2]);
}

void Tensor2Matrices(const Tensor<float>& tensor,
                     arma::field<arma::Mat<float>>& matrices) {
  BOOST_ASSERT(tensor.shape().size() == 3);
  matrices = arma::field<arma::Mat<float>>(tensor[0]);
  const float* data = tensor.data();
  const int mat_size = tensor[1] * tensor[2];
  for (int i = 0; i < tensor[2]; ++i) {
    matrices(i) = arma::Mat<float>(data, tensor[1], tensor[2]);
    data += mat_size;
  }
}

void Tensor2Cubes(const Tensor<float>& tensor,
                  arma::field<arma::Cube<float>>& cubes) {
  BOOST_ASSERT(tensor.shape().size() == 4);
  cubes = arma::field<arma::Cube<float>>(tensor[0]);
  const float* data = tensor.data();
  const int cube_size = tensor[1] * tensor[2] * tensor[3];
  for (int i = 0; i < tensor[0]; ++i) {
    cubes(i) = arma::Cube<float>(data, tensor[1], tensor[2], tensor[3]);
    data += cube_size;
  }
}

} /* end of nn namespace */
} /* end of nerd namespace */
