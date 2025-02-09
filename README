## RabitQ

RabitQ is a fast and memory-efficient implementation for the Approximate Nearest Neighbor (ANN) search problem. It can compress original vectors into a compact binary code (dimension * 1bit) and still achieve high accuracy in ANN search. It was originally designed by Gao, Jianyang and Long, Cheng in 2024. The original paper can be found [here](https://arxiv.org/pdf/2405.12497).

This repo is a C++ implementation of RabitQ. The primary purpose is self-learning, to implement it to ensure understanding of the algorithm and to use it in personal projects. Currently, it's just a naive implementation, not optimized for speed or memory efficiency, but for correctness and readability.

## Prerequisites
- Conan
- C++17
  
  You need to have a compiler that supports C++17 and Conan installed on your machine. You can install Conan by following the instructions [here](https://docs.conan.io/2/installation.html).

## Installation
  Under the root directory of the project, run the following commands:
  ```bash
  conan profile detect --force
  conan install . --output-folder=build --build=missing
  cmake -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=generators/conan_toolchain.cmake -S ${your_project} -B ${your_project}/build/build/Release -G "Unix Makefiles"

  cd build/build/Release
  make -j
  ```
  
## Dataset
  You can download the dataset from [here](http://corpus-texmex.irisa.fr/). Currently, only the siftsmall dataset is tested.
  Extract the dataset and place it in any directory you prefer. Then modify the `main.cpp` file to point to the dataset directory.

## Usage
  After you have successfully compiled the project, you can run the following command to test the RabitQ algorithm under the root directory of the project:

  ```bash
   ./build/build/Release/rabitQ
  ```


