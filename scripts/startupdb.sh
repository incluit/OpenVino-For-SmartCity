#!/bin/bash

if pgrep mongod > /dev/null
then
        echo "mongod is already running"
else
        echo "Attempting to start MongoDB"
        sudo service mongod start
fi
