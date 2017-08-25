#include "NerdNet/util/qt/plot_window.h"

#include <QApplication>
#include <QBrush>
#include <QPen>
#include <QVBoxLayout>
#include <QVector>
#include <armadillo>
#include <boost/assert.hpp>

#include "NerdNet/convert.h"
#include "NerdNet/except/input_exception.h"
#include "ui_plot.h"

namespace nerd {
namespace nn {
namespace ui {

PlotWindow::PlotWindow() : PlotWindow(nullptr) {}

PlotWindow::PlotWindow(QWidget* parent)
    : QMainWindow(parent), ui_(new Ui::PlotUI), plot_(nullptr) {
  ui_->setupUi(this);
  QVBoxLayout* layout = new QVBoxLayout;
  ui_->central_widget->setLayout(layout);
  plot_ = new QCustomPlot(ui_->central_widget);
  layout->addWidget(plot_);
}

PlotWindow::~PlotWindow() {
  if (ui_) delete ui_;
}

void PlotWindow::SetData(
    const Tensor<float>& data_tensor, const Tensor<float>& label_tensor,
    const std::vector<std::pair<Color, Shape>>& label_styles) {
  label_style_ = label_styles;

  arma::Mat<float> data, label;
  Tensor2Matrix(data_tensor, data);
  Tensor2Matrix(label_tensor, label);

  if (data.n_cols != 2) {
    throw except::InputException("Invalid input dimension");
  }

  int num_labeling = label_styles.size();
  for (int l = 0; l < num_labeling; ++l) {
    plot_->addGraph();
  }
  for (int l = 0; l < num_labeling; ++l) {
    plot_->addGraph();
    plot_->graph(l + num_labeling)->setLineStyle(QCPGraph::lsNone);

    int data_size = data.n_rows;
    for (int i = 0; i < data_size; ++i) {
      arma::uvec argmax = arma::sort_index(label.row(i), "descend");
      if (argmax(0) == static_cast<unsigned int>(l)) {
        PlotForeground(plot_->graph(l + num_labeling), data(i, 0), data(i, 1),
                       label_styles[l]);
      }
    }
  }
  plot_->xAxis->setLabel("x");
  plot_->yAxis->setLabel("y");

  float x_min = data.col(0).min(), x_max = data.col(0).max();
  float y_min = data.col(1).min(), y_max = data.col(1).max();
  plot_->xAxis->setRange(x_min, x_max);
  plot_->yAxis->setRange(y_min, y_max);

  arma::Col<float> xx = arma::linspace<arma::Col<float>>(x_min, x_max, 1000);
  arma::Col<float> yy = arma::linspace<arma::Col<float>>(y_min, y_max, 1000);
  BOOST_ASSERT(xx.n_rows == yy.n_rows);

  int span_size = xx.n_rows;
  arma::Mat<float> grid = arma::Mat<float>(span_size * span_size, 2);
  grid.zeros();
  for (int i = 0; i < span_size; ++i) {
    grid.submat(i * span_size, 0, (i + 1) * span_size - 1, 0) = xx;
    for (int j = 0; j < span_size; ++j) {
      grid(i * span_size + j, 1) = yy(i);
    }
  }
  Matrix2Tensor(grid, grid_);
  plot_->replot();

  plot_->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom |
                         QCP::iSelectPlottables);
}

void PlotWindow::SetGridBoundary(const Tensor<float>& prediction) {
  arma::Mat<float> pred, grid;
  Tensor2Matrix(prediction, pred);
  Tensor2Matrix(grid_, grid);

  int num_labeling = label_style_.size();
  for (int l = 0; l < num_labeling; ++l) {
    int data_size = pred.n_rows;
    for (int i = 0; i < data_size; ++i) {
      arma::uvec argmax = arma::sort_index(pred.row(i), "descend");
      if (argmax(0) == static_cast<unsigned int>(l)) {
        PlotBackground(plot_->graph(l), grid(i, 0), grid(i, 1),
                       label_style_[l].first);
      }
    }
  }
}

void PlotWindow::PlotForeground(QCPGraph* graph, double x, double y,
                                std::pair<Color, Shape> label_style) {
  switch (label_style.first) {
    case RED:
      graph->setPen(QPen(QColor(252, 67, 67)));
      break;
    case ORANGE:
      graph->setPen(QPen(QColor(255, 159, 55)));
      break;
    case YELLOW:
      graph->setPen(QPen(QColor(238, 214, 0)));
      break;
    case GREEN:
      graph->setPen(QPen(QColor(128, 199, 84)));
      break;
    case BLUE:
      graph->setPen(QPen(QColor(103, 147, 221)));
      break;
    case PURPLE:
      graph->setPen(QPen(QColor(177, 103, 221)));
      break;
  }
  switch (label_style.second) {
    case DOT:
      graph->setScatterStyle(QCPScatterStyle::ssDot);
      break;
    case CROSS:
      graph->setScatterStyle(QCPScatterStyle::ssCross);
      break;
    case PLUS:
      graph->setScatterStyle(QCPScatterStyle::ssPlus);
      break;
    case CIRCLE:
      graph->setScatterStyle(QCPScatterStyle::ssCircle);
      break;
    case DISC:
      graph->setScatterStyle(QCPScatterStyle::ssDisc);
      break;
    case SQUARE:
      graph->setScatterStyle(QCPScatterStyle::ssSquare);
      break;
    case DIAMOND:
      graph->setScatterStyle(QCPScatterStyle::ssDiamond);
      break;
    case STAR:
      graph->setScatterStyle(QCPScatterStyle::ssStar);
      break;
    case TRIANGLE:
      graph->setScatterStyle(QCPScatterStyle::ssTriangle);
      break;
    case TRIANGLE_INVERTED:
      graph->setScatterStyle(QCPScatterStyle::ssTriangleInverted);
      break;
    case CROSS_SQUARE:
      graph->setScatterStyle(QCPScatterStyle::ssCrossSquare);
      break;
    case PLUS_SQUARE:
      graph->setScatterStyle(QCPScatterStyle::ssPlusSquare);
      break;
    case CROSS_CIRCLE:
      graph->setScatterStyle(QCPScatterStyle::ssCrossCircle);
      break;
    case PLUS_CIRCLE:
      graph->setScatterStyle(QCPScatterStyle::ssPlusCircle);
      break;
    case PEACE:
      graph->setScatterStyle(QCPScatterStyle::ssPeace);
      break;
  }
  graph->addData(x, y);
}

void PlotWindow::PlotBackground(QCPGraph* graph, double x, double y,
                                Color color) {
  switch (color) {
    case RED:
      graph->setPen(QPen(QColor(183, 30, 30)));
      // graph->setBrush(QBrush(QColor(183, 30, 30)));
      break;
    case ORANGE:
      graph->setPen(QPen(QColor(186, 107, 22)));
      // graph->setBrush(QBrush(QColor(186, 107, 22)));
      break;
    case YELLOW:
      graph->setPen(QPen(QColor(196, 163, 8)));
      // graph->setBrush(QBrush(QColor(196, 163, 8)));
      break;
    case GREEN:
      graph->setPen(QPen(QColor(56, 138, 32)));
      // graph->setBrush(QBrush(QColor(56, 138, 32)));
      break;
    case BLUE:
      graph->setPen(QPen(QColor(31, 43, 137)));
      // graph->setBrush(QBrush(QColor(31, 43, 137)));
      break;
    case PURPLE:
      graph->setPen(QPen(QColor(90, 11, 129)));
      // graph->setBrush(QBrush(QColor(90, 11, 129)));
      break;
  }
  graph->setScatterStyle(QCPScatterStyle::ssDisc);
  graph->addData(x, y);
}

} /* end of ui namespace */
} /* end of nn namespace */
} /* end of nerd namespace */
