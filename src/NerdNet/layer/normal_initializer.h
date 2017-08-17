#ifndef RANDOM_NORM_INITIALIZER_H
#define RANDOM_NORM_INITIALIZER_H

#include <random>
#include "NerdNet/layer/variable_initializer.h"

namespace nerd {
namespace nn {

class NormalInitializer : public VariableInitializer {
 public:
  NormalInitializer(float mean, float stddev);
  NormalInitializer(float seed, float mean, float stddev);

  float Next() override;

 private:
  std::mt19937 generator_;
  std::normal_distribution<float> dist_;
};

} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: RANDOM_NORM_INITIALIZER_H */
