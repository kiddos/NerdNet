#ifndef ACT_H
#define ACT_H

#include <math.h>

namespace nn {

struct ActFunc {
  double (*act) (double z);
  double (*actd) (double z);
};

extern ActFunc identity;
extern ActFunc sigmoid;
extern ActFunc arctan;
extern ActFunc tanh;
extern ActFunc bipolarsigmoid;
extern ActFunc lecuntanh;
extern ActFunc hardtanh;
extern ActFunc absolute;
extern ActFunc relucos;
extern ActFunc relusin;
extern ActFunc smoothrectifier;
extern ActFunc logit;
extern ActFunc cos;

} /* end of nn namespace */

#endif /* end of include guard: ACT_H */
