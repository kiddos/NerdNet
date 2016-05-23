#ifndef MOMENTUMTRAINER_H
#define MOMENTUMTRAINER_H

#include "trainer.h"

namespace nn {

class MomentumTrainer : public Trainer {
 public:
  MomentumTrainer();
  MomentumTrainer(NeuralNet& nnet);
  MomentumTrainer(NeuralNet& nnet, double momentum);
  MomentumTrainer(NeuralNet& nnet, double r0, double k, unsigned long step);
  MomentumTrainer(NeuralNet& nnet, double momentum,
                  double r0, double k, unsigned long step);
  MomentumTrainer(const MomentumTrainer& trainer);
  MomentumTrainer& operator= (const MomentumTrainer& trainer);
  ~MomentumTrainer();

  virtual double feeddata(const mat& x, const mat& y, bool ccost=false);

 protected:
  void initmomentum();

  double momentum;
  mat ograd;
  std::vector<mat> hgrads;
};

}

#endif /* end of include guard: MOMENTUMTRAINER_H */
