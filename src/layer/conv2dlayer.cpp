#include "conv2dlayer.h"

namespace nn {

Conv2DLayer::Conv2DLayer() : inputwidth(0), inputheight(0), pnfilter(0),
                             nfilter(0), spatial(0), stride(0), padding(0) {}

Conv2DLayer::Conv2DLayer(int inputwidth, int inputheight, int pnfilter,
                         int nfilter, int spatial, int stride, int padding,
                         double lrate, ActFunc actfunc)
    : Layer(spatial*pnfilter-1, spatial*nfilter, lrate, 0, actfunc),
      inputwidth(inputwidth), inputheight(inputheight),
      inputsize(inputwidth*inputheight), pnfilter(pnfilter), nfilter(nfilter),
      spatial(spatial), stride(stride), padding(padding) {
  randominit(sqrt(spatial*pnfilter));

  outputwidth = (inputwidth - spatial + 2*padding) / stride + 1;
  outputheight = (inputheight - spatial + 2*padding) / stride + 1;
  outputsize = outputwidth * outputheight;
}

Conv2DLayer::Conv2DLayer(const Conv2DLayer& conv)
    : Layer(conv.spatial*conv.pnfilter-1, conv.spatial*conv.nfilter,
            conv.lrate, 0, conv.actfunc) {
  Conv2DLayer::operator= (conv);
}

Conv2DLayer& Conv2DLayer::operator= (const Conv2DLayer& conv) {
  inputwidth = conv.inputwidth;
  inputheight = conv.inputheight;
  pnfilter = conv.pnfilter;
  nfilter = conv.nfilter;
  spatial = conv.spatial;
  stride = conv.spatial;
  padding = conv.padding;
  images = conv.images;
  return *this;
}

#define SPATIAL(m, j, k) \
  m.submat(j*spatial, k*spatial, (j+1)*spatial-1, (k+1)*spatial-1)

mat Conv2DLayer::forwardprop(const mat& pa) {
  a = mat(pa.n_rows, outputsize * nfilter, arma::fill::zeros);
  images = cube(inputheight + 2*padding, inputwidth + 2*padding, pa.n_rows);

  for (uint32_t i = 0 ; i < pa.n_rows ; ++i) {
    // reshape input sample
    mat sample = pa.row(i);
    for (int j = 0 ; j < pnfilter ; ++j) {
      const mat img = toimage(pa, j, inputwidth, inputheight);
      images.slice(i) = addzeropadding(img);

#ifdef HAVE_OPENMP
#pragma omp parallel for default(none) shared(j,i) if (nfilter >= 16)
#endif
      for (int k = 0 ; k < nfilter ; ++k) {
        mat partialoutput;
        convolve(images.slice(i), SPATIAL(W, j, k), partialoutput);
        a.submat(i, k*outputsize, i, (k+1)*outputsize-1) += partialoutput;
      }
    }
  }
  return funcop(a, actfunc.act);
}

mat Conv2DLayer::backprop(const mat& del) {
  delta = mat(del.n_rows, inputsize * pnfilter, arma::fill::zeros);
  grad.zeros();

  mat d = funcop(del, actfunc.actd);

  for (uint32_t i = 0 ; i < d.n_rows ; ++i) {
    mat deltay = d.row(i);

    for (int k = 0 ; k < nfilter ; ++k) {
      const mat dimg = toimage(deltay, k, outputwidth, outputheight);

#ifdef HAVE_OPENMP
#pragma omp parallel for default(none) shared(k,i) if (pnfilter >= 16)
#endif
      for (int j = 0 ; j < pnfilter ; ++j) {
        // compute grad
        mat partialgrad;
        convolve(images.slice(i), dimg, partialgrad);
        partialgrad.reshape(spatial, spatial);
        SPATIAL(grad, j, k) += partialgrad;

        // compute next delta to pass
        const mat paddimg = addzeropadding(dimg);
        mat partialdelta;
        convolve(paddimg, flip(j, k), partialdelta);

        delta.submat(i, j*inputsize, i, (j+1)*inputsize-1) += partialdelta;
      }
    }
  }

  return delta;
}

mat Conv2DLayer::toimage(const mat& pa, int filter, int w, int h) const {
  const int size = w * h;
  mat img = pa.cols(filter*size, (filter+1)*size-1);
  img.reshape(h, w);
  return img;
}

mat Conv2DLayer::addzeropadding(const mat& image) const {
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

mat Conv2DLayer::flip(int pn, int n) const {
  const mat filter = SPATIAL(W, pn, n);
  return arma::fliplr(arma::flipud(filter));
}

#undef SPATIAL

void Conv2DLayer::convolve(const mat& x, const mat& y, mat& output) const {
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
