file(GLOB TENSOR tensor/*.cpp tensor/*.h)
set(CMAKE_INCLUDE_CURRENT_DIR on)
add_library(tensor ${TENSOR})
