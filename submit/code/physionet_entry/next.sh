#! /bin/bash
#
# For example, if invoked as
#    next.sh A00001
# it analyzes record A00001 and (assuming the recording is
# considered to be normal) writes "A00001,N" to answers.txt.

set -e
set -o pipefail

RECORD=$1

python3 main.py -m classify -r $RECORD --dir `pwd`