#!/bin/sh
./preprocess.sh

# Run the original parser
/afs/nd.edu/user14/csesoft/2017-fall/python3.5/bin/python3 cfg.py
/afs/nd.edu/user14/csesoft/2017-fall/python3.5/bin/python3 postprocess.py output/dev.parses > output/dev.parses.post
/afs/nd.edu/user14/csesoft/2017-fall/python3.5/bin/python3 evalb.py output/dev.parses.post trees/dev.trees
