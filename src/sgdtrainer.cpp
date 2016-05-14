#include "sgdtrainer.h"

namespace nn {

SGDTrainer::SGDTrainer() {}

SGDTrainer::SGDTrainer(NeuralNet& nnet) : Trainer(nnet) {}

SGDTrainer::SGDTrainer(NeuralNet& nnet, double r0, double k, unsigned long step)
    : Trainer(nnet, r0, k, step) {}

SGDTrainer::SGDTrainer(const SGDTrainer& trainer)
    : Trainer(*trainer.nnet, trainer.r0, trainer.k, trainer.step) {}

SGDTrainer& SGDTrainer::operator= (const SGDTrainer& trainer) {
  Trainer::operator= (trainer);
  return *this;
}

SGDTrainer::~SGDTrainer() {
  nnet = nullptr;
}

void SGDTrainer::feeddata(const mat& x, const mat& y) {
  iters ++;
  if (usedecay && iters == step) {
    nnet->setlrate(r0 * exp(-k*iters));
  }

  for (uint32_t i = 0 ; i < x.n_rows ; ++i) {
    nnet->forwardprop(x.row(i));
    nnet->backprop(y.row(i));
    nnet->update();
  }
  this->x = x;
  this->y = y;
}

double SGDTrainer::evalcost() const {
  nnet->forwardprop(x);
  nnet->backprop(y);
  return nnet->computecost();
}


}
