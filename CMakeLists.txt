cmake_minimum_required(VERSION 3.20)
project(motion_planning VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MP_BUILD_TESTS "Build the Catch2 test suite" ON)
option(MP_BUILD_EXAMPLES "Build the command-line examples" ON)
option(MP_BUILD_PYTHON "Build the pybind11 Python module" OFF)

include(FetchContent)

# --- Eigen 3.4: system package if present, otherwise header-only download --------------------------------------
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
    message(STATUS "Eigen3 not found on the system; fetching 3.4.0")
    FetchContent_Declare(eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
        SOURCE_SUBDIR do-not-configure)  # headers only; skip Eigen's own CMake project
    FetchContent_MakeAvailable(eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${eigen_SOURCE_DIR}")
endif()

# --- the library ------------------------------------------------------------------------------------------------
add_library(motion_planning
    cpp/src/kinematics.cpp
    cpp/src/pose_control.cpp
    cpp/src/grid.cpp
)
add_library(motion_planning::motion_planning ALIAS motion_planning)
target_include_directories(motion_planning PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>)
target_link_libraries(motion_planning PUBLIC Eigen3::Eigen)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(motion_planning PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(MP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(cpp/tests)
endif()

if(MP_BUILD_EXAMPLES)
    add_subdirectory(cpp/examples)
endif()

if(MP_BUILD_PYTHON)
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_core python/motion_planning/_core.cpp)
    target_link_libraries(_core PRIVATE motion_planning)
    install(TARGETS _core DESTINATION motion_planning)
endif()
