include (CheckIncludeFiles)
check_include_files(omp.h HAVE_OPENMP)

if (OPENMP_FOUND)
  message("--> OpenMP headers found.")
endif (OPENMP_FOUND)
