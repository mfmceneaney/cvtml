#!/bin/bash

$DEPLOY_JAVA_HOME/gradlew run -q -p $DEPLOY_JAVA_HOME --args="`echo " $@"`"
#NOTE: Space is important in " $@" so you don't get empty argument error

#TODO: ABOVE WILL CREATE FILES IN SAME DIRECTORY AS DEPLOY_JAVA_HOME BY DEFAULT.

#java -cp $JCPATH $TORCH_JAVA_OPTS deploy.App $@

#NOTE: TORCH_JAVA_OPTS SHOULD BE set in env.?sh 
