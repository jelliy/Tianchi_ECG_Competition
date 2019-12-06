#! /bin/bash
#
# file: setup.sh
#
# This bash script performs any setup necessary in order to test your
# entry.  It is run only once, before running any other code belonging
# to your entry.

set -e
set -o pipefail

cd packages

pip3 install --no-deps --user PyWavelets-0.5.2.tar.gz
pip3 install --no-deps --user scikit-learn-0.18.1.tar.gz

pip3 install --no-deps --user Theano-0.9.0.tar.gz
pip3 install --no-deps --user Keras-2.0.2.tar.gz
pip3 install --no-deps --user h5py-2.7.0.tar.gz

cd ..