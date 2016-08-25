#include <iostream>
#include <string>
#include <cstdlib>
#include <memory>
#include <time.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "nnet.h"
#include "gradient_checker.h"

using std::shared_ptr;
using std::string;
using std::vector;
using std::endl;
using std::cout;
using nn::Conv2DLayer;
using nn::mat;
using nn::InputLayer;
using nn::NormLayer;
using nn::Layer;
using nn::NeuralNet;
using nn::Trainer;

mat create_sample(int size) {
  mat sample(size, size);
  for (int i = 0 ; i < size ; ++i) {
    for (int j = 0 ; j < size ; ++j) {
      if (j % 2 == i % 2) {
        sample(i, j) = 1;
      } else {
        sample(i, j) = -1;
      }
    }
  }
  return sample;
}

int main(void) {
  const double lrate = 1e-2;
  const double lambda = 0;

  srand(time(nullptr));

  mat sample = create_sample(4);
  sample.reshape(8, 2);

  // cout << sample << endl;

  InputLayer input(2);
  vector<shared_ptr<Layer>> hidden = {
    std::make_shared<Layer>(2, 3, lrate, lambda, nn::sigmoid),
    std::make_shared<NormLayer>(),
  };
  nn::QuadraticOutput output(3, 1, lrate, lambda);
  NeuralNet nnet(input, output, hidden);
  nn::GradientChecker checker(nnet);
  if (!checker.check()) {
    cout << "gradient check failed" << endl;
  }
  // Trainer trainer(nnet);

  // if (!trainer.gradcheck(sample, arma::zeros(8, 1))) {
  //   cout << "gradient check failed" << endl;
  // }

  return 0;
}
