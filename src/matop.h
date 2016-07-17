#include "type.h"

namespace nn {

mat funcop(const mat& m, func f);
mat addcol(const mat& m);
mat addcols(const mat& m, const int index, const double value);
mat addrow(const mat& m);
mat addrows(const mat& m, const int index, const double value);
double sumall(const mat& m);
mat rowsum(const mat& m);
mat exponential(const mat& m);
mat repeat(const mat& m, const int row, const int col);
mat logorithm(const mat& m);

}
