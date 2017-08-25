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
find_package(Boost COMPONENTS system program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_libraries(${Boost_LIBRARIES})

file(GLOB_RECURSE NerdNet_SOURCE
  NerdNet/except/**.h NerdNet/except/**.cc
  NerdNet/layer/**.h NerdNet/layer/**.cc
  NerdNet/trainer/**.h NerdNet/trainer/**.cc
  NerdNet/util/*.h NerdNet/util/*.cc
  NerdNet/*.h
  NerdNet/*.cc)

find_package(Qt5Widgets)
find_package(Qt5PrintSupport)
find_library(QCustomPlot_LIBRARIES qcustomplot)
if (Qt5Widgets_FOUND AND Qt5PrintSupport_FOUND AND QCustomPlot_LIBRARIES)
  set(CMAKE_AUTOMOC ON)
  set(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_RELEASE} -fprofile-arcs -ftest-coverage")

  get_target_property(QtWidget_location Qt5::Widgets LOCATION)
  message(STATUS "QtWidget Version: ${Qt5Widgets_VERSION} at ${QtWidget_location}")

  get_target_property(QtPrintSupport_location Qt5::PrintSupport LOCATION)
  message(STATUS "QtPrintSupport Version: ${Qt5PrintSupport_VERSION} at ${QtPrintSupport_location}")

  message(STATUS "Found QCustomPlot: ${QCustomPlot_LIBRARIES}")
  link_libraries(Qt5::Widgets Qt5::PrintSupport ${QCustomPlot_LIBRARIES})

  qt5_wrap_ui(PLOT_UI_HEADERS NerdNet/util/qt/plot.ui)

  file(GLOB Widget_SOURCE
    NerdNet/util/qt/**.h
    NerdNet/util/qt/**.cc)
  set(NerdNet_SOURCE ${NerdNet_SOURCE} ${Widget_SOURCE}
    ${PLOT_UI_HEADERS})
endif ()

message(STATUS "NerdNet will be build.")
add_library(NerdNet ${NerdNet_SOURCE})
