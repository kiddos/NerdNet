#ifndef NEURALNET_H
#define NEURALNET_H

#include <float.h>
#include <vector>

#include "debug.h"
#include "act.h"
#include "layer.h"
#include "inputlayer.h"
#include "outputlayer.h"

namespace nn {

class NeuralNet {
 public:
  NeuralNet();
  NeuralNet(const NeuralNet& nnet);
  NeuralNet(const InputLayer input, const OutputLayer output,
            std::vector<Layer> layers);
  NeuralNet& operator= (const NeuralNet& nnet);
  void feeddata(const mat x, const mat y, const bool check);
  mat predict(const mat sample);
  double computecost();

  mat getresult() const;
  uint32_t getnumhidden() const;
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


#endif /* end of include guard: NEURALNET_H */


