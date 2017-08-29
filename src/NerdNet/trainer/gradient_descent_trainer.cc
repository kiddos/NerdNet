#include "NerdNet/trainer/gradient_descent_trainer.h"
#include <sstream>

namespace nerd {
namespace nn {
namespace trainer {

GradientDescentTrainer::GradientDescentTrainer(float learning_rate)
    : Trainer(learning_rate) {}

GradientDescentTrainer::GradientDescentTrainer(
    float learning_rate, std::shared_ptr<NerdNet> nerdnet)
    : Trainer(learning_rate, nerdnet) {}

float GradientDescentTrainer::Train(const Tensor<float> data,
                                    const Tensor<float> label) {
  // feed data
  nerdnet_->input_layer()->SetInput(data);
  nerdnet_->cost_function()->SetLabel(label);
  // run
  float loss = nerdnet_->cost_function()->ComputeCost();
  nerdnet_->cost_function()->ComputeDerivative();
  // update weights
  for (int i = 1; i < nerdnet_->layer_count(); ++i) {
    if (nerdnet_->has_variable(i)) {
      nerdnet_->layer(i)->Update(learning_rate_);
    }
  }

  if (verbose_) {
    std::stringstream ss;
    ss << "loss: " << loss / data[0];
    Log(ss.str());
  }
  return loss;
}

} /* end of trainer namespace */
} /* end of nn namespace */
} /* end of nerd namespace */
