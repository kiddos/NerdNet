#ifndef SOFTMAXLAYER_H
#define SOFTMAXLAYER_H

#include "layer.h"

namespace nn {

class SoftmaxLayer : public Layer {
 public:
  SoftmaxLayer();
  SoftmaxLayer(const SoftmaxLayer& softmaxlayer);
  SoftmaxLayer& operator= (const SoftmaxLayer& softmaxlayer);

  virtual mat forwardprop(const mat& pa);
  virtual mat backprop(const mat& delta);
  // override function since there's no weight to update
  virtual void update();
  virtual void update(const mat);
  virtual void randominit(const double);

 private:
  mat p;
};

}

#endif /* end of include guard: SOFTMAXLAYER_H */
