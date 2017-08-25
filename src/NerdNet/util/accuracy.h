#ifndef ACCURACY_H
#define ACCURACY_H

namespace nerd {
namespace nn {

template <typename T>
double Accuracy(const T& prediction, const T& label);

} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: ACCURACY_H */
