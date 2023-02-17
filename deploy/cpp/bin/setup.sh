#!/bin/bash

export PROJECT_DIR=/path/to/cvtml/deploy/cpp
export VENV_PACKAGES=/path/to/venv/lib/python3/site-packages

cd $PROJECT_DIR

# Install PyTorch Scatter
git clone https://github.com/rusty1s/pytorch_scatter.git
pushd pytorch_sparse
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$PROJECT_DIR/libtorch ..
make
mkdir install
cmake --install . --prefix $PWD/install/
popd

# Install PyTorch Sparse
git clone --recurse-submodules https://github.com/rusty1s/pytorch_sparse.git
pushd pytorch_sparse
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$PROJECT_DIR/libtorch;$VENV_PACKAGES/torch" ../
make
mkdir install
cmake --install . --prefix $PWD/install/
popd

# Install PyTorch Spline Conv
git clone https://github.com/rusty1s/pytorch_spline_conv.git
pushd pytorch_sparse
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$PROJECT_DIR/libtorch ..
make
mkdir install
cmake --install . --prefix $PWD/install/
popd

# Install PyTorch Cluster
git clone https://github.com/rusty1s/pytorch_cluster.git
pushd pytorch_sparse
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$PROJECT_DIR/libtorch ..
make
mkdir install
cmake --install . --prefix $PWD/install/
popd

# Compile main project
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$PROJECT_DIR/libtorch;$PROJECT_DIR/pytorch_scatter/build/install;$PROJECT_DIR/pytorch_sparse/build/install" ../
cmake --build . --config Release

echo DONE
