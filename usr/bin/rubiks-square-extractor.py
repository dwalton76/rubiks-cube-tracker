#!/usr/bin/env python2

from rubikssquareextractor import extract_rgb_pixels
import argparse
import json
import logging
import sys

# logging.basicConfig(filename='rubiks.log',
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)12s %(levelname)8s: %(message)s')
log = logging.getLogger(__name__)

# Color the errors and warnings in red
logging.addLevelName(logging.ERROR, "\033[91m   %s\033[0m" % logging.getLevelName(logging.ERROR))
logging.addLevelName(logging.WARNING, "\033[91m %s\033[0m" % logging.getLevelName(logging.WARNING))

if len(sys.argv) > 1:
    target_side = sys.argv[1]
else:
    target_side = None

print(json.dumps(extract_rgb_pixels(target_side), sort_keys=True))
