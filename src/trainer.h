#ifndef TRAINER_H
#define TRAINER_H

#include <vector>
#include <math.h>
#include "neuralnet.h"

namespace nn {

class Trainer {
 public:
  Trainer();
  explicit Trainer(NeuralNet& nnet);
  Trainer(NeuralNet& nnet, double r0, double k, unsigned long step);
  Trainer(const Trainer& trainer);
  Trainer& operator= (const Trainer& trainer);
  ~Trainer();

  virtual bool gradcheck(const mat& x, const mat& y);
  virtual void feeddata(const mat& x, const mat& y);
  virtual double feeddata(const mat& x, const mat& y, bool ccost);

 protected:
  NeuralNet* nnet;
  unsigned long iters, step;
  bool usedecay;
  double k, r0;
};

}

#endif /* end of include guard: TRAINER_H */
