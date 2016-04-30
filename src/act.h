#ifndef ACT_H
#define ACT_H

#include <math.h>

namespace nn {

struct ActFunc {
  double (*act) (double z);
  double (*actd) (double z);
};

ActFunc initactfunc(double (*act)(double), double (*actd)(double));

extern ActFunc identity;
extern ActFunc sigmoid;
extern ActFunc arctan;
extern ActFunc comploglog;
extern ActFunc tanhyperbolic;
extern ActFunc bipolarsigmoid;
extern ActFunc lecuntanh;
extern ActFunc hardtanh;
extern ActFunc absolute;
extern ActFunc relu;
extern ActFunc relucos;
extern ActFunc relusin;
extern ActFunc smoothrelu;

} /* end of nn namespace */

#endif /* end of include guard: ACT_H */
