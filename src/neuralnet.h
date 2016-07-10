#ifndef NEURALNET_H
#define NEURALNET_H

#include <float.h>
#include <vector>
#include <string>
#include <fstream>
#include <memory>

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
            std::vector<std::shared_ptr<Layer>> layers);
  NeuralNet& operator= (const NeuralNet& nnet);

  mat predict(const mat& sample);
  mat forwardprop(const mat& x);
  void backprop(const mat& y);
  void update();
  void update(const mat& ograd, const std::vector<mat>& hgrad);
  bool gradcheck(const mat& x, const mat& y);
  double computecost(const mat& pred, const mat& y);
  void randomize();
  void randomize(uint32_t index);
  void save(const std::string path);
  void load(const std::string path);

  mat getresult() const { return result; };
  uint32_t getnumhidden() const { return hidden.size(); };
  InputLayer getinput() const { return input; };
  Layer gethidden(uint32_t index) const { return *hidden[index]; };
  OutputLayer getoutput() const { return output; };
  void setlrate(double lrate);
  void sethiddenw(uint32_t index, const mat& w) { hidden[index]->setw(w); }
  void setoutputw(const mat& w) { output.setw(w); }

 private:
  bool issame(const mat& m1, const mat& m2);
  double computecost(const mat& x, const mat& y,
                     const mat& perturb, uint32_t idx);
  mat computengrad(const mat& x, const mat& y, int nrows, int ncols, int idx);

  const double eps;
  mat result;
  matfunc cost;
  matfuncd costd;
  InputLayer input;
  std::vector<std::shared_ptr<Layer>> hidden;
  OutputLayer output;
};

}

#endif /* end of include guard: NEURALNET_H */
