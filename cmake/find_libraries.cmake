set(OPENCV_LIBS opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs)
set(MATHGL_LIBS mgl mgl-qt)
set(ARMADILLO_LIBS armadillo)
set(BOOST_LIBS boost_unit_test_framework)

find_library(OPENCV_FOUND NAMES ${OPENCV_LIBS})
find_library(MATHGL_FOUND NAMES ${MATHGL_LIBS})
find_library(ARMADILLO_FOUND NAMES ${ARMADILLO_LIBS})
find_library(BOOST_FOUND NAMES ${BOOST_LIBS})

if (OPENCV_FOUND)
  message("--> OpenCV found.")
endif (OPENCV_FOUND)

if (MATHGL_FOUND)
  message("--> mathgl found.")
endif (MATHGL_FOUND)

if (ARMADILLO_FOUND)
  message("--> armdillo found.")
endif (ARMADILLO_FOUND)

if (BOOST_FOUND)
  message("--> boost found.")
endif (BOOST_FOUND)

if (ARMADILLO_FOUND)
  message("--> Library are all found for building 'nn'")
  set(LIBS_FOUND True)
endif (ARMADILLO_FOUND)

if (OPENCV_FOUND AND MATHGL_FOUND AND ARMADILLO_FOUND)
  message("--> Library are all found for building 'test'")
  set(TEST_LIBS_FOUND True)
  set(EXAMPLE_LIBS_FOUND True)
endif (OPENCV_FOUND AND MATHGL_FOUND AND ARMADILLO_FOUND)
