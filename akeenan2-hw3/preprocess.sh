#!/bin/sh

# Run preprocess and unknown scripts
/afs/nd.edu/user14/csesoft/2017-fall/python3.5/bin/python3 preprocess.py trees/train.trees > trees/train.trees.pre
/afs/nd.edu/user14/csesoft/2017-fall/python3.5/bin/python3 unknown.py trees/train.trees.pre > trees/train.trees.pre.unk
