# Deploy a PyTorch Model in Java

First consider whether it would be easier to do this in C++ or Python.  Then proceed as you see fit.

For an example project of loading a PyTorch Model in Github see the PyTorch [java-demo](https://github.com/pytorch/java-demo).

## Installing Dependencies

You will need to install the C++ PyTorch libraries so you can load the shared libraries.

The basic directions for installing the PyTorch C++ libraries are available on the PyTorch Documentation: [Installing C++ Distributions of PyTorch](https://pytorch.org/cppdocs/installing.html).

You may need additional PyTorch libraries, particularly if you are using [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/).  Most C++ versions of PyTorch Geometric libraries are available on [Matthias Fey's Github](https://github.com/rusty1s).

An example setup script for the PyTorch and PyTorch Geometric libraries is provided in [bin/install_cpp_dependencies.sh](bin/install_cpp_dependencies.sh).  You will need to set
```bash
PROJECT_DIR=/path/to/cvtml/deploy/java
VENV_PACKAGES=/path/to/venv/lib/python3/site-packages
```
to the appropriate directories in this script.

## Getting Started

If you have correctly formatted the [bin/install_cpp_dependencies.sh](bin/install_cpp_dependencies.sh) and you have installed the PyTorch C++ library into this directory, install the PyTorch Geometric dependencies by running
```bash
source bin/setup.sh
```

Once you have installed the C++ dependencies, add the following to your startup script
```bash
pushd /path/to/cvtml/deploy/java/ >> /dev/null
source bin/env.sh
popd >> /dev/null
```
**NOTE** You need to modify the paths in [bin/env.sh](bin/env.sh) if you installed the C++ dependencies somewhere other than this directory.

If your environment is correctly set the `$TORCH_JAVA_OPTS` variable should now have the JVM option for linking all the PyTorch shared libraries you just installed, and the `$DEPLOY_JAVA_HOME` variable should point to this directory.

Now you can build and run the project
```bash
./gradlew build
./gradlew run
```

There should be an example script [bin/run.sh](bin/run.sh) for running the project.  This should allow you to run the project from an arbitrary directory without having to worry about all the gradle arguments.  Try executing it with
```bash
$DEPLOY_JAVA_HOME/bin/run.sh
```
**NOTE** If you modify this project and write files with it, this script may not correctly set the default current working directory for writing files.  As a result files may be written to the project directory.

#

Contact: matthew.mceneaney@duke.edu
