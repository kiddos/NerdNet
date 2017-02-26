#ifndef MAXPOOLLAYER_H
#define MAXPOOLLAYER_H

#include "conv2dlayer.h"

namespace nn {

class MaxPoolLayer : public Conv2DLayer {
 public:
  MaxPoolLayer(int inputwidth, int inputheight, int nfilter,
               int spatial, int stride);
  MaxPoolLayer(const MaxPoolLayer& pool);
  MaxPoolLayer& operator= (const MaxPoolLayer& pool);

  virtual mat forwardprop(const mat& pa);
  virtual mat backprop(const mat& delta);
  virtual void update();
  virtual void update(const mat);

 private:
  void maxpool(const mat& image, mat& output, mat& index);
  double maxval(const mat& partialimage, int& row, int& col);
  mat indexes;

};

}

#endif /* end of include guard: MAXPOOLLAYER_H */
