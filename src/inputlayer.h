#ifndef INPUTLAYER_H
#define INPUTLAYER_H

#include "layer.h"

namespace nn {

class InputLayer : public Layer {
 public:
  InputLayer();
  InputLayer(const InputLayer& input);
  InputLayer(const int innodes);
  virtual InputLayer& operator= (const InputLayer& input);
  virtual mat forwardprop(const mat& input);
};

}

#endif /* end of include guard: INPUTLAYER_H */
