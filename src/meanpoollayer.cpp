#include "meanpoollayer.h"

namespace nn {

MeanPoolLayer::MeanPoolLayer(int inputwidth, int inputheight, int nfilter,
                             int spatial, int stride)
    : Conv2DLayer(inputwidth, inputheight, nfilter, nfilter,
                  spatial, stride, 0, 0, nn::identity) {}

MeanPoolLayer::MeanPoolLayer(const MeanPoolLayer& pool)
    : Conv2DLayer(pool) {}

MeanPoolLayer& MeanPoolLayer::operator= (const MeanPoolLayer& pool) {
  Conv2DLayer::operator= (pool);
  return *this;
}

mat MeanPoolLayer::forwardprop(const mat& pa) {
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
        mat average;

        meanpool(images.slice(i), partialoutput);
        a.submat(i, k*outputsize, i, (k+1)*outputsize-1) += partialoutput;
      }
    }
  }
  return funcop(a, actfunc.act);
}

mat MeanPoolLayer::backprop(const mat& ) {
  return mat();
}

void MeanPoolLayer::meanpool(const mat& image, mat& output) {
  const int ow = (image.n_cols - spatial) / stride + 1;
  const int oh = (image.n_rows - spatial) / stride + 1;
  output = mat(1, ow * oh);
  output.zeros();

  for (int i = 0 ; i < oh ; ++i) {
    for (int j = 0 ; j < ow ; ++j) {
      mat subimage = image.submat(i, j, i+spatial-1, j+spatial-1);
      const double val = meanval(subimage);
      output(0, i*ow+j) = val;
    }
  }
}

double MeanPoolLayer::meanval(const mat& partialimage) {
  double sum = 0;
  for (uint32_t i = 0 ; i < partialimage.n_rows ; ++i) {
    for (uint32_t j = 0 ; j < partialimage.n_cols ; ++j) {
      sum += partialimage(i, j);
    }
  }
  return sum / partialimage.n_rows / partialimage.n_cols;
}

}
