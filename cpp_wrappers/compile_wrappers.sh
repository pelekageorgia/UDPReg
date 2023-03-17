#!/bin/bash

# Compile cpp subsampling
# shellcheck disable=SC2164
cd cpp_subsampling
python3 setup.py build_ext --inplace
cd ..

# Compile cpp neighbors
# shellcheck disable=SC2164
cd cpp_neighbors
python3 setup.py build_ext --inplace
cd ..