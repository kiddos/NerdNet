#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <QApplication>
#include <armadillo>
#include <boost/program_options.hpp>

#include "NerdNet/convert.h"
#include "NerdNet/layers.h"
#include "NerdNet/nerd_net.h"
#include "NerdNet/trainer/gradient_descent_trainer.h"
#include "NerdNet/util.h"
#include "NerdNet/util/qt/plot_window.h"

using nerd::nn::Tensor;
using nerd::nn::VariableShape;
using nerd::nn::NerdNet;
using nerd::nn::FCLayer;
using nerd::nn::ReluLayer;
using nerd::nn::SoftmaxCrossEntropy;
using nerd::nn::trainer::GradientDescentTrainer;
using nerd::nn::ui::PlotWindow;
using arma::Mat;
using arma::Row;

void GenerateData(Mat<float>& data, Mat<float>& label, int data_size,
                  float noise_stddev) {
  data = arma::Mat<float>(data_size, 2);
  label = arma::Mat<float>(data_size, 2);
  data.zeros();
  label.zeros();

  std::mt19937 gen(
      std::chrono::system_clock::now().time_since_epoch().count());
  std::normal_distribution<float> noise_dist(0.0f, noise_stddev);
  for (int i = 0; i < data_size / 4; ++i) {
    data(i, 0) = noise_dist(gen);
    data(i, 1) = noise_dist(gen);
    label(i, 0) = 1.0f;
  }

  for (int i = data_size / 4; i < data_size / 2; ++i) {
    data(i, 0) = 1 + noise_dist(gen);
    data(i, 1) = noise_dist(gen);
    label(i, 1) = 1.0f;
  }

  for (int i = data_size / 2; i < 3 * data_size / 4; ++i) {
    data(i, 0) = noise_dist(gen);
    data(i, 1) = 1 + noise_dist(gen);
    label(i, 1) = 1.0f;
  }

  for (int i = 3 * data_size / 4; i < data_size; ++i) {
    data(i, 0) = 1 + noise_dist(gen);
    data(i, 1) = 1 + noise_dist(gen);
    label(i, 0) = 1.0f;
  }
}

int main(int argc, char* argv[]) {
  using namespace boost::program_options;
  options_description desc("Classic Exclusive Or Problem");
  desc.add_options()("help,h", "help messages");
  desc.add_options()("data-size", value<int>()->default_value(100),
                     "data size to plot and classify");
  desc.add_options()("noise", value<float>()->default_value(0.1),
                     "noise in data");
  desc.add_options()("hidden-size", value<int>()->default_value(16),
                     "hidden layer size");
  desc.add_options()("learning-rate", value<float>()->default_value(1e-3),
                     "learning rate to train");
  desc.add_options()("max-epoch", value<int>()->default_value(50000),
                     "max epoch to train");

  variables_map vmap;
  store(parse_command_line(argc, argv, desc), vmap);
  notify(vmap);

  if (vmap.count("help")) {
    std::cout << desc << "\n";
    return 0;
  }

  int data_size = vmap["data-size"].as<int>();
  float noise = vmap["noise"].as<float>();
  int hidden_size = vmap["hidden-size"].as<int>();
  float learning_rate = vmap["learning-rate"].as<float>();
  int max_epoch = vmap["max-epoch"].as<int>();

  Mat<float> data, label;
  GenerateData(data, label, data_size, noise);

  std::shared_ptr<NerdNet> nerdnet = std::make_shared<NerdNet>();
  nerdnet->AddLayer<FCLayer>(VariableShape{2, hidden_size});
  nerdnet->AddLayer<ReluLayer>();
  nerdnet->AddLayer<SoftmaxCrossEntropy>(VariableShape{hidden_size, 2});

  Tensor<float> data_tensor, label_tensor;
  Matrix2Tensor(data, data_tensor);
  Matrix2Tensor(label, label_tensor);
  GradientDescentTrainer trainer(learning_rate, nerdnet);
  for (int epoch = 0; epoch < max_epoch + 1; ++epoch) {
    float loss = trainer.Train(data_tensor, label_tensor);
    if (epoch % 1000 == 0) {
      Tensor<float> prediction_tensor = nerdnet->Feed(data_tensor);
      std::cout << "\r" << epoch << ". loss: " << loss << ", accuracy: "
                << Accuracy(prediction_tensor, label_tensor) * 100.0 << "%"
                << std::flush;
    }
  }
  std::cout << std::endl;

  QApplication app(argc, argv);
  PlotWindow window;

  window.SetData(data_tensor, label_tensor,
                 {{PlotWindow::RED, PlotWindow::DISC},
                  {PlotWindow::BLUE, PlotWindow::SQUARE}});
  Tensor<float> grid = window.grid();
  Tensor<float> prediction = nerdnet->Feed(grid);

  window.SetGridBoundary(prediction);

  window.show();
  return app.exec();
}
