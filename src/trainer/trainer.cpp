#include "trainer.h"

namespace nn {

Trainer::Trainer() : nnet(nullptr), iters(0), step(0), usedecay(false), k(0), r0(0) {}

Trainer::Trainer(NeuralNet& nnet) : nnet(&nnet), iters(0), step(0),
                                    usedecay(false), k(0), r0(0) {}

Trainer::Trainer(NeuralNet& nnet, double r0, double k, unsigned long step)
    : nnet(&nnet), iters(0), step(step), usedecay(true), k(k), r0(r0) {}

Trainer::Trainer(const Trainer& trainer)
    : nnet(trainer.nnet), iters(trainer.iters), step(trainer.step),
      usedecay(trainer.usedecay), k(trainer.k), r0(trainer.r0) {}

Trainer& Trainer::operator= (const Trainer& trainer) {
  nnet = trainer.nnet;
  iters = trainer.iters;
  step = trainer.step;
  usedecay = trainer.usedecay;
  k = trainer.k;
  r0 = trainer.r0;
  return *this;
}

Trainer::~Trainer() {
  nnet = nullptr;
}

void Trainer::feeddata(const mat& x, const mat& y) {
  iters ++;
  if (usedecay && iters == step) {
    nnet->setlrate(r0 * exp(-k*iters));
  }

  nnet->forwardprop(x);
  nnet->backprop(y);
  nnet->update();
}

double Trainer::feeddata(const mat& x, const mat& y, bool ccost) {
  feeddata(x, y);

  double cost = 0;
  if (ccost) {
    cost = nnet->computecost(nnet->getresult(), y);
  }
  return cost;
}

}
