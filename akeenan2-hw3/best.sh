#!/bin/sh
./preprocess.sh

# Generate the fallback file
/afs/nd.edu/user14/csesoft/2017-fall/python3.5/bin/python3 cfg.py --strings strings/test.strings --output output/test.parses
/afs/nd.edu/user14/csesoft/2017-fall/python3.5/bin/python3 postprocess.py output/test.parses > output/test.parses.post
/afs/nd.edu/user14/csesoft/2017-fall/python3.5/bin/python3 cfg_mods.py -m 1 --strings strings/test.strings --output output/test.parses.mod1 --fallback output/test.parses.post
/afs/nd.edu/user14/csesoft/2017-fall/python3.5/bin/python3 cfg_mods.py -m 2 --strings strings/test.strings --output output/test.parses.mod2 --fallback output/test.parses.mod1.post
/afs/nd.edu/user14/csesoft/2017-fall/python3.5/bin/python3 cfg_mods.py -m 3 --strings strings/test.strings --output output/test.parses.mod3 --fallback output/test.parses.mod2.post

# Run the modification
/afs/nd.edu/user14/csesoft/2017-fall/python3.5/bin/python3 cfg_mods.py -m 4 --strings strings/test.strings --output output/test.parses.best --fallback output/test.parses.mod3.post
/afs/nd.edu/user14/csesoft/2017-fall/python3.5/bin/python3 evalb.py output/test.parses.best.post trees/test.trees
