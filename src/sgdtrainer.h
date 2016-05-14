#ifndef SGDTRAINER_H
#define SGDTRAINER_H

#include "trainer.h"

namespace nn {

class SGDTrainer : public Trainer {
 public:
  SGDTrainer();
  explicit SGDTrainer(NeuralNet& nnet);
  SGDTrainer(NeuralNet& nnet, double r0, double k, unsigned long step);
  SGDTrainer(const SGDTrainer& trainer);
  SGDTrainer& operator= (const SGDTrainer& trainer);
  ~SGDTrainer();

  virtual void feeddata(const mat& x, const mat& y);
  virtual double evalcost() const;

 protected:
  mat x, y;
};

}

#endif /* end of include guard: SGDTRAINER_H */
