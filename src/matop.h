#include "type.h"

namespace nn {

mat funcop(const mat m, func f);
mat addcol(const mat m);
mat addcols(const mat m, const int index, const double value);
mat addrow(const mat m);
mat addrows(const mat m, const int index, const double value);
double sumall(const mat m);
mat normalize(const mat& m);

}
