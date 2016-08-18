find_program(CLANG_FOUND clang clang++)

if (CLANG_FOUND)
  set(CMAKE_CXX_COMPILER clang++)
else (CLANG_FOUND)
  set(CMAKE_CXX_COMPILER g++)
endif (CLANG_FOUND)
