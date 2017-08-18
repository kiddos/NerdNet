#include "NerdNet/trainer/trainer.h"
#include <ctime>
#include <iostream>

namespace nerd {
namespace nn {
namespace trainer {

Trainer::Trainer(float learning_rate)
    : learning_rate_(learning_rate), verbose_(false) {}

Trainer::Trainer(float learning_rate, std::shared_ptr<NerdNet> nerdnet)
    : learning_rate_(learning_rate), verbose_(false), nerdnet_(nerdnet) {
  Log("Initializing NerdNet...\n");
  if (!nerdnet_->Init()) {
    Error("Fail to initialize Nerdnet\n");
  }
  Log("NerdNet initialized\n");
}

std::string Trainer::CurrentTime() {
  char buffer[128];
  std::time_t raw_time;
  std::time(&raw_time);
  struct tm* time_info = std::localtime(&raw_time);
  std::strftime(buffer, 128, "%D-%T", time_info);
  return std::string(buffer);
}

void Trainer::Log(const std::string& msg) {
  if (verbose_) {
    std::cout << "\r[" << CurrentTime() << "] " << msg << std::flush;
  }
}

void Trainer::Warning(const std::string& msg) {
  if (verbose_) {
    std::cout << "\r[\033[1;31m" << CurrentTime() << "\033[0m] " << msg
              << std::flush;
  }
}

void Trainer::Error(const std::string& msg) {
  if (verbose_) {
    std::cout << "\r[\033[1;33m" << CurrentTime() << "\033[0m] " << msg
              << std::flush;
  }
}

} /* end of trainer namespace */
} /* end of nn namespace */
} /* end of nerd namespace */
