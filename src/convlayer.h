#ifndef CONLAYER_H
#define CONLAYER_H

#include "layer.h"

namespace nn {

class ConvLayer : public Layer {
 public:
  ConvLayer();
  ConvLayer(const int inputwidth, const int inputheight,
            const int pnfilter, const int nfilter,
            const int spatial, const int stride, const int padding,
            const double lrate, func act, func actd);
  ConvLayer(const int inputwidth, const int inputheight,
            const int pnfilter, const int nfilter,
            const int spatial, const int stride, const int padding,
            const double lrate, ActFunc actfunc);
  ConvLayer(const ConvLayer& conv);
  ConvLayer& operator= (const ConvLayer& conv);
  virtual mat forwardprop(const mat pa);
  virtual mat backprop(const mat delta);

  int getnfilter() const;
  int getspatialsize() const;
  int getstride() const;
  int getzeropadding() const;


 protected:
  mat toimage(const mat& pa, int pnfilter) const;
  mat addzeropadding(const mat& image) const;
  mat flip(int pn, int n) const;
  void convole(const mat& x, const mat& y, mat& output) const;

  int inputwidth, inputheight, inputsize;
  int pnfilter, nfilter, spatial, stride, padding;
  mat padinput;
};

}

#endif /* end of include guard: CONLAYER_H */

