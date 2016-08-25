#ifndef GRADIENT_CHECKER_H
#define GRADIENT_CHECKER_H

#include "neuralnet.h"

namespace nn {

class GradientChecker {
 public:
  explicit GradientChecker(const NeuralNet& net);

  bool check();
  static bool issame(const mat& m1, const mat& m2);

 private:
  NeuralNet nnet;
  mat random_data, random_label;
};

}

#endif /* end of include guard: GRADIENT_CHECKER_H */
