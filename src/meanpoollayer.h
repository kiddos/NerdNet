#ifndef MEANPOOLLAYER_H
#define MEANPOOLLAYER_H

#include "conv2dlayer.h"

namespace nn {

class MeanPoolLayer : public Conv2DLayer {
 public:
  MeanPoolLayer(int inputwidth, int inputheight, int nfilter,
               int spatial, int stride);
  MeanPoolLayer(const MeanPoolLayer& pool);
  MeanPoolLayer& operator= (const MeanPoolLayer& pool);

  virtual mat forwardprop(const mat& pa);
  virtual mat backprop(const mat& delta);
  virtual void update();
  virtual void update(const mat);

 private:
  void meanpool(const mat& image, mat& output, mat& index);
  double meanval(const mat& partialimage);
  mat averages;

};

}

#endif /* end of include guard: MEANPOOLLAYER_H */
