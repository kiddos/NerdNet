#ifndef PLOT_WINDOW_H
#define PLOT_WINDOW_H

#include <utility>
#include <vector>

#include <qcustomplot.h>
#include <QMainWindow>

#include "NerdNet/tensor.h"

namespace Ui {

class PlotUI;

} /* end of Ui namespace */

namespace nerd {
namespace nn {
namespace ui {

class PlotWindow : public QMainWindow {
  Q_OBJECT

 public:
  enum Color { RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE };
  enum Shape {
    DOT,
    CROSS,
    PLUS,
    CIRCLE,
    DISC,
    SQUARE,
    DIAMOND,
    STAR,
    TRIANGLE,
    TRIANGLE_INVERTED,
    CROSS_SQUARE,
    PLUS_SQUARE,
    CROSS_CIRCLE,
    PLUS_CIRCLE,
    PEACE
  };
  PlotWindow();
  PlotWindow(QWidget* parent);
  ~PlotWindow();

  Tensor<float> grid() const { return grid_; }

  void SetData(const Tensor<float>& data_tensor,
               const Tensor<float>& label_tensor,
               const std::vector<std::pair<Color, Shape>>& label_style_);
  void SetGridBoundary(const Tensor<float>& prediction);

 private:
  void PlotForeground(QCPGraph* graph, double x, double y,
                      std::pair<Color, Shape> label_style);
  void PlotBackground(QCPGraph* graph, double x, double y, Color color);

  Ui::PlotUI* ui_;
  QCustomPlot* plot_;

  std::vector<std::pair<Color, Shape>> label_style_;
  Tensor<float> grid_;
};

} /* end of ui namespace */
} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: PLOT_WINDOW_H */
