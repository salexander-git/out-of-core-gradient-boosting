cmake_minimum_required(VERSION 3.15)
project(out_of_core_gradient_boosting LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# pybind11 installed with pip: locate its CMake package through the interpreter.
find_package(Python 3 REQUIRED COMPONENTS Interpreter Development.Module)
if(NOT pybind11_DIR)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Python extension: gb_core
pybind11_add_module(gb_core engine_wrapper.cpp)
target_include_directories(gb_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(gb_core PRIVATE -Wall)
target_link_libraries(gb_core PRIVATE OpenMP::OpenMP_CXX)

# Native inference CLI: gb_engine
add_executable(gb_engine main.cpp)
target_include_directories(gb_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(gb_engine PRIVATE -Wall)
target_link_libraries(gb_engine PRIVATE OpenMP::OpenMP_CXX)
