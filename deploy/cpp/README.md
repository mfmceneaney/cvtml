# Deploy a PyTorch Model in C++

Use [CMake](https://cmake.org) to build your C++ project.  This is by far the easiest way to integrate the needed dependencies for deploying a PyTorch Model.  The essentials are available on the Pytorch documentation: [Loading a Torchscript Model in C++](https://pytorch.org/tutorials/advanced/cpp_export.html).

The [PyTorch C++ Api](https://pytorch.org/cppdocs/) has its own documentation too.

## Installing Dependencies

The basic directions for installing the PyTorch C++ libraries are available on the PyTorch Documentation: [Installing C++ Distributions of PyTorch](https://pytorch.org/cppdocs/installing.html).

You may need additional PyTorch libraries, particularly if you are using [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/).

Most C++ versions of PyTorch Geometric libraries are available on [Matthias Fey's Github](https://github.com/rusty1s).

An example setup script for external libraries is provided in [bin/setup.sh](bin/setup.sh).

#

Contact: matthew.mceneaney@duke.edu
