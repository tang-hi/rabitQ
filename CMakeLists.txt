cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(rabitQ VERSION 0.0.1 LANGUAGES CXX)

find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)


include_directories(${CMAKE_SOURCE_DIR}/include)
add_executable(rabitQ src/main.cpp src/rabitQ.cpp)
target_compile_options(rabitQ PRIVATE -O3 -g -mavx2 -march=native -fopenmp)
target_link_options(rabitQ PRIVATE -lgomp)

target_link_libraries(rabitQ Eigen3::Eigen spdlog::spdlog)