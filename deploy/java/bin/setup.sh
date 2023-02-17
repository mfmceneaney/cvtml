#!/bin/bash

export DEPLOY_JAVA_HOME=$PWD
echo "DEPLOY_JAVA_HOME=$PWD" | grep DEPLOY_JAVA_HOME --color=auto

./bin/install_cpp_dependencies.sh

# Build groovy library
./gradlew build
