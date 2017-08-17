#include "NerdNet/layer/normal_initializer.h"

#include <chrono>
#include <random>

namespace nerd {
namespace nn {

NormalInitializer::NormalInitializer(float mean, float stddev)
    : generator_(std::chrono::system_clock::now().time_since_epoch().count()),
      dist_(mean, stddev) {}

NormalInitializer::NormalInitializer(float seed, float mean, float stddev)
    : generator_(seed), dist_(mean, stddev) {}

float NormalInitializer::Next() { return dist_(generator_); }

} /* end of nn namespace */
} /* end of nerd namespace */
