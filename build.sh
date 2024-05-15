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

docker-compose up

