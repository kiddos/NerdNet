#ifndef NORMLAYER_H
#define NORMLAYER_H

#include "layer.h"

namespace nn {

class NormLayer : public Layer {
 public:
  NormLayer();
  NormLayer(const NormLayer& normlayer);
  NormLayer& operator= (const NormLayer& normlayer);

  virtual mat forwardprop(const mat& pa);
  virtual mat backprop(const mat& d);
  // override function since there's no weight to update
  virtual void update();
  virtual void update(const mat);
  virtual void randominit(const double);

 private:
  double normval;
};

}

#endif /* end of include guard: NORMLAYER_H */
