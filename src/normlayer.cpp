#include "normlayer.h"

#include <iostream>
using namespace std;

namespace nn {

NormLayer::NormLayer() {
  lrate = 0;
  lambda = 0;
  act = nn::identity.act;
  actd = nn::identity.actd;
}

NormLayer::NormLayer(const NormLayer& normlayer)
    : Layer(normlayer.getpnnodes(), normlayer.getw().n_cols,
            normlayer.getlrate(), normlayer.getlambda(),
            normlayer.getact(), normlayer.getactd()) {}

NormLayer& NormLayer::operator= (const NormLayer& normlayer) {
  Layer::operator= (normlayer);
  return *this;
}

mat NormLayer::forwardprop(const mat& pa) {
  this->pa = pa;
  const double maxval = pa.max();
  const double minval = pa.min();
  if (normval == 0)
    normval = maxval - minval;
  z = pa / normval;
  a = z;
  return a;
}

mat NormLayer::backprop(const mat& d) {
  const mat result = d / normval;
  normval = 0;
  return result;
}

}
