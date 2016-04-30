#ifndef TYPES_H
#define TYPES_H

#ifndef LIBMAT
#include <armadillo>
#include <stdlib.h>
#include <time.h>
namespace nn {
  typedef arma::mat mat;
}
#else
#include "mat.h"
namespace nn {
  typedef core::mat<double> mat;
}
#endif

namespace nn {
  typedef double (*func)(double);
  typedef mat (*matfunc)(mat,mat);
  typedef mat (*matfuncd)(mat,mat,mat);
}

#endif /* end of include guard: TYPES_H */
