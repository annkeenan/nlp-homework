#!/bin/sh
./preprocess.sh

# Generate the fallback file
/afs/nd.edu/user14/csesoft/2017-fall/python3.5/bin/python3 cfg.py
/afs/nd.edu/user14/csesoft/2017-fall/python3.5/bin/python3 postprocess.py output/dev.parses > output/dev.parses.post

# Run the modification
/afs/nd.edu/user14/csesoft/2017-fall/python3.5/bin/python3 cfg_mods.py -m 1 --output output/dev.parses.mod1 --fallback output/dev.parses.post
/afs/nd.edu/user14/csesoft/2017-fall/python3.5/bin/python3 evalb.py output/dev.parses.mod1.post trees/dev.trees
