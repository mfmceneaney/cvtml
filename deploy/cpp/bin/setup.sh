#!/bin/bash

export PROJECT_DIR=/work/clas12/users/mfmce/cmake_torch_test_2_13_23
export VENV_PACKAGES=/work/clas12/users/mfmce/venv_ifarm/lib/python3.9/site-packages

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
