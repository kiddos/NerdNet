#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>

#include "gradient_checker.h"
#include "log.h"

namespace nn {

GradientChecker::GradientChecker(const NeuralNet& net) {
  srand(time(nullptr));

  nnet = net;
  int input_dim = net.getinput().getnnodes();
  int output_dim = net.getoutput().getnnodes();

  LOG("input dimesion: %d\n", input_dim);
  LOG("output dimesion: %d\n", output_dim);

  random_data = mat(1, input_dim);
  random_label = mat(1, output_dim);

  for (int i = 0 ; i < input_dim ; ++i) {
    random_data[i] = (rand() % 10000) / 10000.0;
  }

  for (int i = 0 ; i < output_dim ; ++i) {
    random_label[i] = (rand() % 10000) / 10000.0;
  }
}

bool GradientChecker::check() {
  const double eps = 1e-6;

  LOG("gradient check starting...");
  bool success = true;
  for (int i = 0 ; i < static_cast<int>(nnet.getnumhidden()) ; ++i) {
    const mat W = nnet.gethidden(i).getw();
    mat perturb = W;
    mat gradient = nnet.gethidden(i).getgrad();

    LOG("checking layer %i\n", i);

    for (int j = 0 ; j < static_cast<int>(perturb.n_rows) ; ++j) {
      for (int k = 0 ; k < static_cast<int>(perturb.n_cols) ; ++k) {
        LOG("checking layer weights %llu\n", j * perturb.n_cols + k);

        // compute first cost
        perturb[j * perturb.n_cols + k] += eps;
        nnet.sethiddenw(i, perturb);
        const mat prediction1 = nnet.forwardprop(random_data);
        const double loss1 = nnet.computecost(prediction1, random_label);

        // compute second cost
        perturb[j * perturb.n_cols + k] -= 2 * eps;
        nnet.sethiddenw(i, perturb);
        const mat prediction2 = nnet.forwardprop(random_data);
        const double loss2 = nnet.computecost(prediction2, random_label);

        // reset Weights
        nnet.sethiddenw(i, W);
        gradient[j * gradient.n_cols + k] = (loss1 - loss2) / (2 * eps);
      }
    }

    nnet.forwardprop(random_data);
    nnet.backprop(random_label);

    const mat target = nnet.gethidden(i).getgrad();
    if (issame(gradient, target)) {
      LOG("gradient check for layer %d passed.\n", i);
    } else {
      LOG("FAILED!! gradient check for layer %d failed\n.", i);
      success = false;
    }
  }
  return success;
}

bool GradientChecker::issame(const mat& m1, const mat& m2) {
  if (m1.n_rows != m2.n_rows || m1.n_cols != m2.n_cols) {
    return false;
  }

  bool same = true;
  for (int i = 0 ; i < static_cast<int>(m1.n_rows) ; ++i) {
    for (int j = 0 ; j < static_cast<int>(m1.n_cols) ; ++j) {
      const double val1 = m1[i * m1.n_cols + j];
      const double val2 = m2[i * m1.n_cols + j];
      if (fabs(val1 - val2) < 1e-6) {
        same = false;
      }
    }
  }
  if (!same) {
    std::cout << m1 << '\n';
    std::cout << m2 << '\n';
  }
  return same;
}

}
