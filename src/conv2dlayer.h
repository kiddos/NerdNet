#ifndef CONLAYER_H
#define CONLAYER_H

#include "layer.h"

namespace nn {

class Conv2DLayer : public Layer {
 public:
  Conv2DLayer();
  Conv2DLayer(int inputwidth, int inputheight, int pnfilter, int nfilter,
              int spatial, int stride, int padding,
              double lrate, ActFunc actfunc);
  Conv2DLayer(const Conv2DLayer& conv);
  Conv2DLayer& operator= (const Conv2DLayer& conv);
  virtual mat forwardprop(const mat& pa);
  virtual mat backprop(const mat& delta);

  int getnfilter() const { return nfilter; }
  int getspatialsize() const { return spatial; }
  int getstride() const { return stride; }
  int getzeropadding() const { return padding; }

  int inputwidth, inputheight, inputsize;
  int outputwidth, outputheight, outputsize;

 protected:
  mat toimage(const mat& pa, int filter, int w, int h) const;
  mat addzeropadding(const mat& image) const;
  mat flip(int pn, int n) const;

  int pnfilter, nfilter, spatial, stride, padding;
  cube images;

 private:
  void convolve(const mat& x, const mat& y, mat& output) const;

};

}

#endif /* end of include guard: CONLAYER_H */
