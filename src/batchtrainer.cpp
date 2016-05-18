#include "batchtrainer.h"

namespace nn {

BatchTrainer::BatchTrainer() {}

BatchTrainer::BatchTrainer(NeuralNet& nnet, int batchsize)
    : Trainer(nnet), batchsize(batchsize) {}

BatchTrainer::BatchTrainer(NeuralNet& nnet, int batchsize,
                           double r0, double k, unsigned long step)
    : Trainer(nnet, r0, k, step), batchsize(batchsize) {}

BatchTrainer::BatchTrainer(const BatchTrainer& trainer)
    : Trainer(*trainer.nnet, trainer.r0, trainer.k, trainer.step),
      batchsize(trainer.batchsize) {}

BatchTrainer& BatchTrainer::operator= (const BatchTrainer& trainer) {
  Trainer::operator= (trainer);
  batchsize = trainer.batchsize;
  return *this;
}

BatchTrainer::~BatchTrainer() {
  nnet = nullptr;
}

void BatchTrainer::feeddata(const mat& x, const mat& y) {
  this->x = x;
  this->y = y;

  mat trainx = x;
  mat trainy = y;
  trainx.insert_rows(trainx.n_rows, x.submat(0, 0, batchsize-2, x.n_cols-1));
  trainy.insert_rows(trainy.n_rows, y.submat(0, 0, batchsize-2, y.n_cols-1));

  for (uint32_t i = 0 ; i < x.n_rows ; i += batchsize) {
    const int start = i;
    const int end = i + batchsize - 1;
    nnet->forwardprop(trainx.rows(start, end));
    nnet->backprop(trainy.rows(start, end));
    nnet->update();

    iters ++;
  }

  if (usedecay && iters == step) {
    nnet->setlrate(r0 * exp(-k*iters));
  }
}

double BatchTrainer::evalcost() const {
  nnet->forwardprop(x);
  nnet->backprop(y);
  return nnet->computecost();
}

}
