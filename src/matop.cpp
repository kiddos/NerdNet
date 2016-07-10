#include "matop.h"

namespace nn {

mat funcop(const mat m, func f) {
  mat newmat = m;
  newmat.transform(f);
  return newmat;
}

mat addcol(const mat m) {
  mat newmat(m.n_rows, m.n_cols+1, arma::fill::ones);
  newmat.submat(0, 1, newmat.n_rows-1, newmat.n_cols-1) = m;
  return newmat;
}

mat addcols(const mat m, const int index, const double value) {
  mat newmat = m;
  mat values;
  if (value == 0) {
    values = mat(m.n_rows, 1);
    values.zeros();
  } else if (value == 1) {
    values = mat(m.n_rows, 1);
    values.ones();
  } else {
    values = mat(m.n_rows, 1);
    values.ones();
    values = values * value;
  }
  newmat.insert_cols(index, values);
  return newmat;
}

mat addrow(const mat m) {
  mat newmat = m;
  mat values(1, m.n_cols, arma::fill::ones);
  newmat.insert_rows(0, values);
  return newmat;
}

mat addrows(const mat m, const int index, const double value) {
  mat newmat = m;
  mat values;
  if (value == 0) {
    values = mat(1, m.n_cols);
    values.zeros();
  } else if (value == 1) {
    values = mat(1, m.n_cols);
    values.ones();
  } else {
    values = mat(1, m.n_cols);
    values.ones();
    values = values * value;
  }
  newmat.insert_rows(index, values);
  return newmat;
}

double sumall(const mat m) {
  const mat temp = m;
  return arma::accu(temp);
}

mat normalize(const mat& m) {
  return arma::normalise(m);
}

mat norm(const mat& m) {
  return arma::sqrt(arma::sum(m % m, 1));
}

}
