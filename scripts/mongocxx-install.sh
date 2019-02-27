#!/bin/bash

sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 9DA31620334BD75D9DCB49F368818C72E52529D4
 echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu xenial/mongodb-org/4.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-4.0.list
sudo apt-get update
sudo apt-get install -y mongodb-org
sudo apt-get install libmongoc-dev
cd
wget https://github.com/mongodb/mongo-c-driver/releases/download/1.13.1/mongo-c-driver-1.13.1.tar.gz
tar xzf mongo-c-driver-1.13.1.tar.gz
cd mongo-c-driver-1.13.1/
mkdir cmake-build
cd cmake-build/
cmake -DENABLE_AUTOMATIC_INIT_AND_CLEANUP=OFF ..
make
sudo make install
cd ..
git clone https://github.com/mongodb/mongo-cxx-driver.git     --branch releases/stable --depth 1
cd mongo-cxx-driver/build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local ..
sudo make EP_mnmlstc_core
make && sudo make install