# Deploy a PyTorch Model in Java

First consider whether it would be easier to do this in C++ or Python.  Then proceed as you see fit.

For an example project of loading a PyTorch Model in Github see the PyTorch [java-demo](https://github.com/pytorch/java-demo).

## Installing Dependencies

You will need to install the C++ PyTorch libraries so you can load the shared libraries.

The basic directions for installing the PyTorch C++ libraries are available on the PyTorch Documentation: [Installing C++ Distributions of PyTorch](https://pytorch.org/cppdocs/installing.html).

You may need additional PyTorch libraries, particularly if you are using [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/).  Most C++ versions of PyTorch Geometric libraries are available on [Matthias Fey's Github](https://github.com/rusty1s).

An example setup script for external libraries is provided in [bin/setup.sh](bin/setup.sh).  You will need to set 
```bash
PROJECT_DIR=/path/to/this/project
VENV_PACKAGES=/path/to/venv/lib/python3/site-packages
```
to the appropriate directories in this script.

#

Contact: matthew.mceneaney@duke.edu
