#include "momentumtrainer.h"

namespace nn {

MomentumTrainer::MomentumTrainer() : momentum(0.6) {}

MomentumTrainer::MomentumTrainer(NeuralNet& nnet, double momentum)
    : Trainer(nnet), momentum(momentum) {
  initmomentum();
}

MomentumTrainer::MomentumTrainer(NeuralNet& nnet, double momentum,
                  double r0, double k, unsigned long step)
    : Trainer(nnet, r0, k, step), momentum(momentum) {
  initmomentum();
}


MomentumTrainer::MomentumTrainer(const MomentumTrainer& trainer)
    : Trainer(*trainer.nnet, trainer.r0, trainer.k, trainer.step),
      momentum(trainer.momentum), ograd(trainer.ograd), hgrads(trainer.hgrads) {}

MomentumTrainer& MomentumTrainer::operator= (const MomentumTrainer& trainer) {
  Trainer::operator= (trainer);
  momentum = trainer.momentum;
  ograd = trainer.ograd;
  hgrads = trainer.hgrads;
  return *this;
}

MomentumTrainer::~MomentumTrainer() {
  nnet = nullptr;
}

void MomentumTrainer::initmomentum() {
  ograd = nnet->getoutput().getw();
  ograd.zeros();

  const int nhidden = nnet->getnumhidden();
  for (int i = 0 ; i < nhidden ; ++i) {
    mat m = nnet->gethidden(i).getw();
    m.zeros();
    hgrads.push_back(m);
  }
}

void MomentumTrainer::feeddata(const mat& x, const mat& y) {
  iters ++;
  if (usedecay && iters == step) {
    nnet->setlrate(r0 * exp(-k*iters));
  }

  nnet->forwardprop(x);
  nnet->backprop(y);
  ograd = momentum * ograd + (1.0-momentum) * nnet->getoutput().getgrad();
  for (uint32_t i = 0 ; i < hgrads.size() ; ++i) {
    hgrads[i] = momentum * hgrads[i] + (1.0-momentum) * nnet->gethidden(i).getgrad();
  }

  nnet->update(ograd, hgrads);
}

}
