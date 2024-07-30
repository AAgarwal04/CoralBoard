#!/bin/bash

pip3 install joblib
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential gfortran libatlas-base-dev
sudo pip3 install --upgrade pip setuptools wheel
sudo pip3 install numpy

sudo pip3 install scikit-learn
