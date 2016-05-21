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

  outputwidth = (inputwidth - spatial + 2*padding) / stride + 1;
  outputheight = (inputheight - spatial + 2*padding) / stride + 1;
  outputsize = outputwidth * outputheight;
}

ConvLayer::ConvLayer(const int inputwidth, const int inputheight,
                     const int pnfilter, const int nfilter,
                     const int spatial, const int stride, const int padding,
                     const double lrate, ActFunc actfunc)
    : ConvLayer(inputwidth, inputheight, pnfilter, nfilter,
                spatial, stride, padding, lrate,
                actfunc.act, actfunc.actd) {
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

mat ConvLayer::forwardprop(const mat& pa) {
  mat output(pa.n_rows, outputsize * nfilter, arma::fill::zeros);
  images = cube(inputheight + 2*padding, inputwidth + 2*padding, pa.n_rows);

  for (uint32_t i = 0 ; i < pa.n_rows ; ++i) {
    // reshape input sample
    mat sample = pa.row(i);
    for (int j = 0 ; j < pnfilter ; ++j) {
      const mat img = toimage(pa, j, inputwidth, inputheight);
      images.slice(i) = addzeropadding(img);

#ifdef HAVE_OPENMP
#pragma omp parallel for default(none) shared(output,j,i) if (nfilter >= 16)
#endif
      for (int k = 0 ; k < nfilter ; ++k) {
        mat partialoutput;
        convolve(images.slice(i), SPATIAL(W, j, k), partialoutput);
        output.submat(i, k*outputsize, i, (k+1)*outputsize-1) += partialoutput;
      }
    }
  }
  return funcop(output, act);
}

mat ConvLayer::backprop(const mat& delta) {
  mat nextdelta(delta.n_rows, inputsize * pnfilter, arma::fill::zeros);
  grad.zeros();

  for (uint32_t i = 0 ; i < delta.n_rows ; ++i) {
    mat deltay = pa.row(i);

    for (int k = 0 ; k < nfilter ; ++k) {
      mat dimg = toimage(deltay, k, outputwidth, outputheight);

#ifdef HAVE_OPENMP
#pragma omp parallel for default(none) shared(nextdelta,dimg,k,i) if (pnfilter >= 16)
#endif
      for (int j = 0 ; j < pnfilter ; ++j) {
        // compute grad
        mat partialgrad;
        convolve(images.slice(i), dimg, partialgrad);
        partialgrad.reshape(spatial, spatial);
        SPATIAL(grad, j, k) += partialgrad;

        // compute next delta to pass
        mat partialdelta;
        convolve(dimg, SPATIAL(W, j, k), partialdelta);
        nextdelta.submat(i, j*inputsize, i, (j+1)*inputsize-1) += partialdelta;
      }
    }
  }

  return nextdelta;
}

mat ConvLayer::toimage(const mat& pa, int filter, int w, int h) const {
  const int size = w * h;
  mat img = pa.cols(filter*size, (filter+1)*size-1);
  img.reshape(h, w);
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

#undef SPATIAL

void ConvLayer::convolve(const mat& x, const mat& y, mat& output) const {
  const int ow = (x.n_cols - y.n_cols) / stride + 1;
  const int oh = (x.n_rows - y.n_rows) / stride + 1;
  output = mat(1, ow * oh);

  for (int i = 0 ; i < oh ; i+=stride) {
    for (int j = 0 ; j < ow ; j+=stride) {
      const mat result = x.submat(i, j, i+y.n_rows-1, j+y.n_cols-1) % y;
      const double val = sumall(result);
      output(0, i*ow+j) = val;
    }
  }
}

}
