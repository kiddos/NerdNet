#include "NerdNet/trainer/sgd_trainer.h"

#include <sstream>

#include "NerdNet/convert.h"

namespace nerd {
namespace nn {
namespace trainer {

SGDTrainer::SGDTrainer(float learning_rate) : Trainer(learning_rate) {}

SGDTrainer::SGDTrainer(float learning_rate, std::shared_ptr<NerdNet> nerdnet)
    : Trainer(learning_rate, nerdnet) {}

float SGDTrainer::Train(const Tensor<float> data, const Tensor<float> label) {
  arma::Mat<float> d, l;
  Tensor2Matrix(data, d);
  Tensor2Matrix(label, l);

  float loss = 0;
  int data_size = d.n_rows;
  for (int i = 0; i < data_size; ++i) {
    nerdnet_->input_layer()->SetInput(d.row(i));
    nerdnet_->cost_function()->SetLabel(l.row(i));

    loss += nerdnet_->cost_function()->ComputeCost();

    nerdnet_->cost_function()->ComputeDerivative();
    // update weights
    for (int i = 1; i < nerdnet_->layer_count(); ++i) {
      nerdnet_->layer(i)->Update(learning_rate_);
    }
  }

  loss /= static_cast<float>(data_size);
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
