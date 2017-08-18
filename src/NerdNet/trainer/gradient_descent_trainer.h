#ifndef GRADIENT_DESCENT_TRAINER_H
#define GRADIENT_DESCENT_TRAINER_H

#include "NerdNet/trainer/trainer.h"

namespace nerd {
namespace nn {
namespace trainer {

class GradientDescentTrainer : public Trainer {
 public:
  GradientDescentTrainer(float learning_rate);
  GradientDescentTrainer(float learning_rate,
                         std::shared_ptr<NerdNet> nerdnet);

  float Train(const Tensor<float> data, const Tensor<float> label) override;
};

} /* end of trainer namespace */
} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: GRADIENT_DESCENT_TRAINER_H */
