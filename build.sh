#!/bin/bash

mkdir -p assets/iris

function file_exists() {
    if [ -e "$1" ]; then
        return 0
    else
        return 1
    fi
}

if ! file_exists "assets/iris/iris.data"; then
    dataset_url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    wget "$dataset_url" -O "iris.data"
    mv "iris.data" assets/iris/
fi

mkdir -p build
cmake -G Ninja -S . -B build -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DLIBCXX_BUILD=/home/hope/Sources/llvm-project/build
ninja -C build
build/flower_neural_network