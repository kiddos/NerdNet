set(CMAKE_INCLUDE_CURRENT_DIR ON)

# armadillo linear algebra
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
message(STATUS "Armadillo Library Version: ${ARMADILLO_VERSION_STRING}")
link_libraries(${ARMADILLO_LIBRARIES})

# OpenBLAS for performance
find_library(OpenBLAS openblas)
if (NOT OpenBLAS)
  message(WARNING "OpenBLAS not installed")
else ()
  message(STATUS "using OpenBLAS ${OpenBLAS}")
endif ()
link_libraries(${OpenBLAS})

# boost util
## Boost
find_package(Boost COMPONENTS system REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_libraries(${Boost_LIBRARIES})

file(GLOB_RECURSE NerdNet_SOURCE
  NerdNet/except/**.h NerdNet/except/**.cc
  NerdNet/layer/**.h NerdNet/layer/**.cc
  NerdNet/trainer/**.h NerdNet/trainer/**.cc
  NerdNet/util/*.h NerdNet/util/*.cc
  NerdNet/convert.*
  NerdNet/nerd_net.*
  NerdNet/tensor.*
  NerdNet/variable_shape.*
  *.cpp
  *.h)

find_package(OpenCV 3.0 COMPONENTS core highgui shape)
if (OpenCV_FOUND)
  message(STATUS "Using OpenCV")
  link_libraries(${OpenCV_LIBS})

  message(STATUS "Building Util...")
  file(GLOB Util_SOURCE
    NerdNet/util.*)
  set(NerdNet_SOURCE ${NerdNet_SOURCE} ${Util_SOURCE})
endif ()

find_package(Qt5Widgets)
if (Qt5Widgets_FOUND)
  set(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_RELEASE} -fprofile-arcs -ftest-coverage")
  get_target_property(QtWidget_location Qt5::Widgets LOCATION)

  message(STATUS "QtWidget Version: ${Qt5Widgets_VERSION} at ${QtWidget_location}")

  link_libraries(Qt5::Widgets)
  set(CMAKE_AUTOMOC ON)

  file(GLOB Widget_SOURCE
    NerdNet/util/qt/**.h
    NerdNet/util/qt/**.cc)
  set(NerdNet_SOURCE ${NerdNet_SOURCE} ${Widget_SOURCE})
endif ()

message(STATUS "NerdNet will be build.")
add_library(NerdNet ${NerdNet_SOURCE})
