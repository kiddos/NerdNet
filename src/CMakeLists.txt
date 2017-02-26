file(GLOB TENSOR tensor/*.cpp tensor/*.h)
file(GLOB TENSOR_OP tensor/ops/*.cpp tensor/ops/*.h)
set(CMAKE_INCLUDE_CURRENT_DIR on)
add_library(tensor ${TENSOR} ${TENSOR_OP})
