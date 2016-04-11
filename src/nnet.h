#ifndef NNET_H
#define NNET_H

#include <float.h>
#include <vector>
#include "type.h"
#include "layer.h"

namespace nn {

class NeuralNet {
 public:
  NeuralNet(const InputLayer input, const OutputLayer output,
            std::vector<Layer> layers,
            mat (*cost)(mat,mat),
            mat (*costd)(mat,mat), bool gradcheck);
  ~NeuralNet();
  void feeddata(const mat x, const mat y);
  mat predict(const mat sample);
  void gradcheck();

 private:
  double computecost(const mat perturb, const uint32_t idx);
  mat computengrad(const int nrows, const int ncols, const int idx);

  const double eps;
  bool check;
  mat x, y;
  mat result;
  mat (*cost)(mat,mat);
  mat (*costd)(mat,mat);
  InputLayer input;
  std::vector<Layer> hidden;
  OutputLayer output;
};

}

#endif /* end of include guard: NNET_H */

