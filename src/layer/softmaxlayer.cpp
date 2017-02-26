#include "softmaxlayer.h"

namespace nn {

SoftmaxLayer::SoftmaxLayer() : Layer(0, 0, 0, 0, nn::identity) {}

SoftmaxLayer::SoftmaxLayer(const SoftmaxLayer& layer) : Layer(layer) {}

SoftmaxLayer& SoftmaxLayer::operator= (const SoftmaxLayer& layer) {
  Layer::operator= (layer);
  p = layer.p;
  return *this;
}

mat SoftmaxLayer::forwardprop(const mat& pa) {
  const mat expo = exponential(pa);
  const mat expo_sum = repeat(rowsum(expo), 1, expo.n_cols);
  p = expo / expo_sum;
  return p;
}

mat SoftmaxLayer::backprop(const mat& delta) {
  const mat expo = exponential(delta);
  const mat expo_sum = repeat(rowsum(expo), 1, expo.n_cols);
  return expo / expo_sum;
}

void SoftmaxLayer::update() {}

void SoftmaxLayer::update(const mat) {}

void SoftmaxLayer::randominit(const double) {}

}
