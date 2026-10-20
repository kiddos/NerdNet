# Find OpenCV (optional for examples)
find_package(OpenCV QUIET)

if(OpenCV_FOUND)
  add_executable(spiral spiral.cpp)
  target_link_libraries(spiral PRIVATE nnet ${OpenCV_LIBS})
  target_include_directories(spiral PRIVATE ${OpenCV_INCLUDE_DIRS})

  add_executable(xor_problem xor_problem.cpp)
  target_link_libraries(xor_problem PRIVATE nnet ${OpenCV_LIBS})
  target_include_directories(xor_problem PRIVATE ${OpenCV_INCLUDE_DIRS})
else()
  message(STATUS "OpenCV not found. Skipping building OpenCV-dependent examples (spiral, xor_problem).")
endif()
