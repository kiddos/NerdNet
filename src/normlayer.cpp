#include "normlayer.h"

#include <iostream>
using namespace std;

namespace nn {

NormLayer::NormLayer() : Layer(0, 0, 0, 0, nn::identity) {}

NormLayer::NormLayer(const NormLayer& layer) : Layer(layer) {}

NormLayer& NormLayer::operator= (const NormLayer& layer) {
  Layer::operator= (layer);
  normval = layer.normval;
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
