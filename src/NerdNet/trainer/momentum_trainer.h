#ifndef MOMENTUM_TRAINER_H
#define MOMENTUM_TRAINER_H

#include "NerdNet/trainer/trainer.h"

namespace nerd {
namespace nn {
namespace trainer {

class MomentumTrainer : public Trainer {
 public:
  MomentumTrainer(float learning_rate, float momentum);
  MomentumTrainer(float learning_rate, float momentum,
                  std::shared_ptr<NerdNet> nerdnet);

  float Train(const Tensor<float> data, const Tensor<float> label) override;

 private:
  std::vector<arma::Mat<float>> wgrads_, bgrads_;
  float momentum_;
};

} /* end of trainer namespace */
} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: MOMENTUM_TRAINER_H */
