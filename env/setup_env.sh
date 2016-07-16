#!/bin/bash

# Download miniconda installer
curl https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh > /tmp/Miniconda2-latest-Linux-x86_64.sh

# Run miniconda installer
bash /tmp/Miniconda2-latest-Linux-x86_64.sh

# Use conda to install the required packages
conda install numpy pandas matplotlib scikit-learn

# and keras, too
pip install keras

 