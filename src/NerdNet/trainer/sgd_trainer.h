#ifndef SGD_TRAINER_H
#define SGD_TRAINER_H

#include "NerdNet/trainer/trainer.h"

namespace nerd {
namespace nn {
namespace trainer {

class SGDTrainer : public Trainer {
 public:
  SGDTrainer(float learning_rate);
  SGDTrainer(float learning_rate, std::shared_ptr<NerdNet> nerdnet);

  float Train(const Tensor<float> data, const Tensor<float> label) override;
};

} /* end of trainer namespace */
} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: SGD_TRAINER_H */
