#ifndef POOLLAYER_H
#define POOLLAYER_H

#include "layer.h"
#include "conv2dlayer.h"

namespace nn {

class PoolLayer : Conv2DLayer {
 public:
  PoolLayer(int inputwidth, int inputheight, int nfilter,
            int spatial, int stride, double lrate, func act, func actd);
  PoolLayer(const PoolLayer& poollayer);
  PoolLayer& operator= (const PoolLayer& poollayer);

  virtual mat forwardprop(const mat& pa);
  virtual mat backprop(const mat& delta);
};

}

#endif /* end of include guard: POOLLAYER_H */
