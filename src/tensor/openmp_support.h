#include <omp.h>

#ifdef _OPENMP
#define PARALLEL_FOR() _Pragma("omp parallel for")
#else
#define PARALLEL_FOR()
#endif
