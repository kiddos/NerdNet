#ifndef CONLAYER_H
#define CONLAYER_H

#include "layer.h"

namespace nn {

class ConvLayer : public Layer {
 public:
  ConvLayer();
  ConvLayer(const int nfilter, const int spatial,
            const int stride, const int padding,
            const double lrate, double (*act)(double), double (*actd)(double));
  ConvLayer(const int nfilter, const int spatial,
            const int stride, const int padding,
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
  int nfilter, spatial, stride, padding;
};

}

#endif /* end of include guard: CONLAYER_H */

