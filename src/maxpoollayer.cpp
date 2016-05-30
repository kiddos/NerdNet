#include "maxpoollayer.h"

namespace nn {

MaxPoolLayer::MaxPoolLayer(int inputwidth, int inputheight, int nfilter,
                           int spatial, int stride)
    : Conv2DLayer(inputwidth, inputheight, nfilter, nfilter,
                  spatial, stride, 0, 0, nn::identity) {}

MaxPoolLayer::MaxPoolLayer(const MaxPoolLayer& pool)
    : Conv2DLayer(pool) {
  indexes = pool.indexes;
}

MaxPoolLayer& MaxPoolLayer::operator= (const MaxPoolLayer& pool) {
  Conv2DLayer::operator= (pool);
  indexes = pool.indexes;
  return *this;
}

mat MaxPoolLayer::forwardprop(const mat& pa) {
  a = mat(pa.n_rows, outputsize * nfilter, arma::fill::zeros);
  images = cube(inputheight + 2*padding, inputwidth + 2*padding, pa.n_rows);
  indexes = mat(pa.n_rows, outputsize*2);

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
        mat index;

        maxpool(images.slice(i), partialoutput, index);
        a.submat(i, k*outputsize, i, (k+1)*outputsize-1) += partialoutput;
        indexes.submat(i, 0, i, outputsize*2-1) = index;
      }
    }
  }
  return funcop(a, act);
}

mat MaxPoolLayer::backprop(const mat& delta) {
  return delta;
}

void MaxPoolLayer::update() {}

void MaxPoolLayer::update(const mat) {}

void MaxPoolLayer::maxpool(const mat& image, mat& output, mat& index) {
  const int ow = (image.n_cols - spatial) / stride + 1;
  const int oh = (image.n_rows - spatial) / stride + 1;
  output = mat(1, ow * oh);
  output.zeros();
  index = mat(1, ow * oh * 2);
  index.zeros();

  for (int i = 0 ; i < oh ; ++i) {
    for (int j = 0 ; j < ow ; ++j) {
      mat subimage = image.submat(i, j, i+spatial-1, j+spatial-1);
      int row = 0, col = 0;
      const double val = maxval(subimage, row, col);
      output(0, i*ow+j) = val;
      index(0, i*ow*2+j) = row;
      index(0, i*ow*2+j+1) = col;
    }
  }
}

double MaxPoolLayer::maxval(const mat& partialimage, int& row, int& col) {
  double maxval = 0;
  for (uint32_t i = 0 ; i < partialimage.n_rows ; ++i) {
    for (uint32_t j = 0 ; j < partialimage.n_cols ; ++j) {
      if (partialimage(i, j) > maxval) {
        maxval = partialimage(i, j);
        row = i;
        col = j;
      }
    }
  }
  return maxval;
}

}
