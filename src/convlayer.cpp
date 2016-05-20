#include "convlayer.h"

namespace nn {

ConvLayer::ConvLayer() : inputwidth(0), inputheight(0), nfilter(0),
                         spatial(0), stride(0), padding(0) {}

ConvLayer::ConvLayer(const int inputwidth, const int inputheight,
                     const int pnfilter, const int nfilter,
                     const int spatial, const int stride, const int padding,
                     const double lrate, func act, func actd)
    : Layer(spatial*pnfilter-1, spatial*nfilter, lrate, 0, act, actd),
      inputwidth(inputwidth), inputheight(inputheight),
      inputsize(inputwidth*inputheight), nfilter(nfilter),
      spatial(spatial), stride(stride), padding(padding) {
  randominit(sqrt(spatial*pnfilter));
}

ConvLayer::ConvLayer(const int inputwidth, const int inputheight,
                     const int pnfilter, const int nfilter,
                     const int spatial, const int stride, const int padding,
                     const double lrate, ActFunc actfunc)
    : Layer(spatial*pnfilter-1, spatial*nfilter, lrate, 0, actfunc),
      inputwidth(inputwidth), inputheight(inputheight),
      inputsize(inputwidth*inputheight), nfilter(nfilter),
      spatial(spatial), stride(stride), padding(padding) {
  randominit(sqrt(spatial*pnfilter));
}

ConvLayer::ConvLayer(const ConvLayer& conv) {
  ConvLayer::operator= (conv);
}

ConvLayer& ConvLayer::operator= (const ConvLayer& conv) {
  inputwidth = conv.inputwidth;
  inputheight = conv.inputheight;
  nfilter = conv.nfilter;
  spatial = conv.spatial;
  stride = conv.spatial;
  padding = conv.padding;
  return *this;
}

mat ConvLayer::image(const mat& pa, const int pnfilter) const {
  mat img = pa.cols(pnfilter*inputsize, (pnfilter+1)*inputsize-1);
  img.reshape(inputheight, inputwidth);
  return img;
}

#define FILTER(j, k) \
  W.submat(j*spatial, k*spatial, (j+1)*spatial-1, (k+1)*spatial-1);

mat ConvLayer::forwardprop(const mat pa) {
  const uint32_t pnfilter = W.n_rows / spatial;
  const int outputwidth = (inputwidth - spatial + 2*padding) / stride + 1;
  const int outputheight = (inputheight - spatial + 2*padding) / stride + 1;
  const int outputsize = outputwidth * outputheight;
  mat output(pa.n_rows, outputsize * nfilter, arma::fill::zeros);
  for (uint32_t i = 0 ; i < pa.n_rows ; ++i) {
    // reshape input sample
    mat sample = pa.row(i);
    for (uint32_t j = 0 ; j < pnfilter ; ++j) {
      mat img = image(pa, j);
      addzeropadding(img);

#ifdef HAVE_OPENMP
  #pragma omp parallel for default(none) shared(output, img) if (outputsize>1048576)
#endif
      for (int k = 0 ; k < nfilter ; ++k) {
        for (int l = 0 ; l < inputheight ; l+=stride) {
          for (int m = 0 ; m < inputwidth ; m+=stride) {
            mat result = img.submat(l, m, l+spatial-1, m+spatial-1) % FILTER(j, k);
            const double val = sumall(result);
            output(i, k*outputsize+l*outputheight+m) += val;
          }
        }
      }
    }
  }
  return output;
}


}
