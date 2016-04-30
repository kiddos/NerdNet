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
            std::vector<Layer> layers);
  void feeddata(const mat x, const mat y, const bool check);
  mat predict(const mat sample);
  double computecost();
  mat getresult() const;
  InputLayer getinput() const;
  Layer gethidden(const uint32_t index) const;
  OutputLayer getoutput() const;
  void setlrate(const double lrate);

 private:
  bool issame(const mat m1, const mat m2);
  bool gradcheck();
  double computecost(const mat perturb, const uint32_t idx);
  mat computengrad(const int nrows, const int ncols, const int idx);

  const double eps;
  mat x, y;
  mat result;
  matfunc cost;
  matfuncd costd;
  InputLayer input;
  std::vector<Layer> hidden;
  OutputLayer output;
};

}

#endif /* end of include guard: NNET_H */

