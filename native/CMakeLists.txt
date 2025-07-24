cmake_minimum_required(VERSION 3.20)
project(rinexpy_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Optimisation flags. Match what numba/llvm would generate.
if(MSVC)
    add_compile_options(/O2 /DNDEBUG)
else()
    add_compile_options(-O3 -DNDEBUG -Wall -Wextra -Wpedantic)
endif()

# Locate Python and nanobind via the build-time helpers scikit-build-core
# wires up for us.
find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)

# nanobind ships its own CMake config; ask Python where it is.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${NANOBIND_CMAKE_DIR}")
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
    _ext
    STABLE_ABI                        # build against Python 3.12 stable ABI
    src/decode_obs_batch.cpp
    src/crc24q.cpp
    src/lagrange_sp3.cpp
    src/lambda_ils.cpp
    src/msm_decode.cpp
    src/nav_subframes.cpp
    src/bindings.cpp
)

target_compile_definitions(_ext PRIVATE
    NB_DOMAIN=rinexpy_native
)

# Install the compiled extension inside the rinexpy_native package dir.
install(TARGETS _ext LIBRARY DESTINATION rinexpy_native)
