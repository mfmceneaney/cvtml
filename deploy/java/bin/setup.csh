#!/bin/csh

setenv DEPLOY_JAVA_HOME $PWD
echo "DEPLOY_JAVA_HOME=$PWD" | grep DEPLOY_JAVA_HOME --color=auto

# Build groovy library
./gradlew build
