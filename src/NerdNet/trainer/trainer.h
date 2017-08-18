#ifndef TRAINER_H
#define TRAINER_H

#include "NerdNet/nerd_net.h"

#include <string>
#include <memory>

namespace nerd {
namespace nn {
namespace trainer {

class Trainer {
 public:
  Trainer(float learning_rate);
  Trainer(float learning_rate, std::shared_ptr<NerdNet> nerdnet);
  virtual ~Trainer() {}

  void set_learning_rate(float learning_rate) {
    learning_rate_ = learning_rate;
  }
  void set_verbose(bool verbose) { verbose_ = verbose; }
  void set_nerdnet(std::shared_ptr<NerdNet> nerdnet) {
    nerdnet_ = nerdnet;

    Log("Initializing NerdNet...\n");
    if (!nerdnet_->Init()) {
      Error("Fail to initialize Nerdnet\n");
    }
    Log("NerdNet initialized\n");
  }

  virtual float Train(const Tensor<float> data, const Tensor<float> label) = 0;

 protected:
  std::string CurrentTime();
  void Log(const std::string& msg);
  void Warning(const std::string& msg);
  void Error(const std::string& msg);

  float learning_rate_;
  bool verbose_;
  std::shared_ptr<NerdNet> nerdnet_;
};

} /* end of trainer namespace */
} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: TRAINER_H */
