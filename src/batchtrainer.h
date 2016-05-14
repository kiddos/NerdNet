#ifndef BATCHTRAINER_H
#define BATCHTRAINER_H

#include "trainer.h"

namespace nn {

class BatchTrainer : public Trainer {
 public:
  BatchTrainer();
  BatchTrainer(NeuralNet& nnet, int batchsize);
  BatchTrainer(NeuralNet& nnet, int batchsize,
               double r0, double k, unsigned long step);
  BatchTrainer(const BatchTrainer& trainer);
  BatchTrainer& operator= (const BatchTrainer& trainer);
  ~BatchTrainer();

  virtual void feeddata(const mat& x, const mat& y);

 protected:
  mat x, y;
  int batchsize;
};

}

#endif /* end of include guard: BATCHTRAINER_H */
