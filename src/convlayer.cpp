#include "convlayer.h"

namespace nn {

ConvLayer::ConvLayer() : inputwidth(0), inputheight(0), pnfilter(0), nfilter(0),
                         spatial(0), stride(0), padding(0) {}

ConvLayer::ConvLayer(const int inputwidth, const int inputheight,
                     const int pnfilter, const int nfilter,
                     const int spatial, const int stride, const int padding,
                     const double lrate, func act, func actd)
    : Layer(spatial*pnfilter-1, spatial*nfilter, lrate, 0, act, actd),
      inputwidth(inputwidth), inputheight(inputheight),
      inputsize(inputwidth*inputheight), pnfilter(pnfilter), nfilter(nfilter),
      spatial(spatial), stride(stride), padding(padding) {
  randominit(sqrt(spatial*pnfilter));
}

ConvLayer::ConvLayer(const int inputwidth, const int inputheight,
                     const int pnfilter, const int nfilter,
                     const int spatial, const int stride, const int padding,
                     const double lrate, ActFunc actfunc)
    : Layer(spatial*pnfilter-1, spatial*nfilter, lrate, 0, actfunc),
      inputwidth(inputwidth), inputheight(inputheight),
      inputsize(inputwidth*inputheight), pnfilter(pnfilter), nfilter(nfilter),
      spatial(spatial), stride(stride), padding(padding) {
  randominit(sqrt(spatial*pnfilter));
}

ConvLayer::ConvLayer(const ConvLayer& conv)
    : Layer(conv.spatial*conv.pnfilter-1, conv.spatial*conv.nfilter,
            conv.lrate, 0, conv.act, conv.actd) {
  ConvLayer::operator= (conv);
}

ConvLayer& ConvLayer::operator= (const ConvLayer& conv) {
  inputwidth = conv.inputwidth;
  inputheight = conv.inputheight;
  pnfilter = conv.pnfilter;
  nfilter = conv.nfilter;
  spatial = conv.spatial;
  stride = conv.spatial;
  padding = conv.padding;
  return *this;
}

#define SPATIAL(m, j, k) \
  m.submat(j*spatial, k*spatial, (j+1)*spatial-1, (k+1)*spatial-1)

mat ConvLayer::forwardprop(const mat pa) {
  const int outputwidth = (inputwidth - spatial + 2*padding) / stride + 1;
  const int outputheight = (inputheight - spatial + 2*padding) / stride + 1;
  const int outputsize = outputwidth * outputheight;
  mat output(pa.n_rows, outputsize * nfilter, arma::fill::zeros);

  for (uint32_t i = 0 ; i < pa.n_rows ; ++i) {
    // reshape input sample
    mat sample = pa.row(i);
    for (int j = 0 ; j < pnfilter ; ++j) {
      mat img = toimage(pa, j);
      img = addzeropadding(img);

#ifdef HAVE_OPENMP
#pragma omp parallel for default(none) shared(output,img,j,i) if (nfilter >= 16)
#endif
      for (int k = 0 ; k < nfilter ; ++k) {
        mat partialoutput;
        convole(img, SPATIAL(W, j, k), partialoutput);
        output.submat(i, k*outputsize, i, (k+1)*outputsize-1) += partialoutput;
      }
    }
  }
  return funcop(output, act);
}


mat ConvLayer::backprop(const mat) {
  return mat();
}

mat ConvLayer::toimage(const mat& pa, int pnfilter) const {
  mat img = pa.cols(pnfilter*inputsize, (pnfilter+1)*inputsize-1);
  img.reshape(inputheight, inputwidth);
  return img;
}


mat ConvLayer::addzeropadding(const mat& image) const {
  mat newimage = image;
  for (int i = 0 ; i < padding ; ++i) {
    newimage = addrows(newimage, 0, 0);
    newimage = addrows(newimage, newimage.n_rows, 0);
  }
  for (int i = 0 ; i < padding ; ++i) {
    newimage = addcols(newimage, 0, 0);
    newimage = addcols(newimage, newimage.n_cols, 0);
  }
  return newimage;
}

mat ConvLayer::flip(int pn, int n) const {
  const mat filter = SPATIAL(W, pn, n);
  return arma::fliplr(arma::flipud(filter));
}

void ConvLayer::convole(const mat& x, const mat& y, mat& output) const {
  const int outputwidth = (x.n_cols - y.n_cols) / stride + 1;
  const int outputheight = (x.n_rows - y.n_rows) / stride + 1;
  output = mat(1, outputwidth * outputheight);

  for (int i = 0 ; i < outputheight ; i+=stride) {
    for (int j = 0 ; j < outputwidth ; j+=stride) {
      const mat result = x.submat(i, j, i+y.n_rows-1, j+y.n_cols-1) % y;
      const double val = sumall(result);
      output(0, i*outputwidth+j) = val;
    }
  }
}

#undef FILTER

}
