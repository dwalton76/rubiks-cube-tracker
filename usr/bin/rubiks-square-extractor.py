#!/usr/bin/env python2

from rubikssquareextractor import RubiksVideo, RubiksImage, merge_two_dicts
import argparse
import json
import logging
import os
import sys


# Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)12s %(levelname)8s: %(message)s')
log = logging.getLogger(__name__)

# Color the errors and warnings in red
logging.addLevelName(logging.ERROR, "\033[91m   %s\033[0m" % logging.getLevelName(logging.ERROR))
logging.addLevelName(logging.WARNING, "\033[91m %s\033[0m" % logging.getLevelName(logging.WARNING))

# Command line args
parser = argparse.ArgumentParser("Rubiks Square Extractor")
parser.add_argument('-d', '--directory', type=str, help='Directory of images to examine')
parser.add_argument('-f', '--filename', type=str, help='Image to examine')
parser.add_argument('-w', '--webcam', action='store_true', help='examine webcam')
args = parser.parse_args()

if args.webcam:
    rvid = RubiksVideo()
    rvid.analyze_webcam()

elif args.filename:
    rimg = RubiksImage(debug=True)
    rimg.analyze_file(args.filename)
    print(json.dumps(rimg.data, sort_keys=True))

else:
    data = {}

    if not os.path.isdir(args.directory):
        print "ERROR: directory %s does not exist" % args.directory
        sys.exit(1)

    for (side_index, side_name) in enumerate(('U', 'L', 'F', 'R', 'B', 'D')):
        filename = os.path.join(args.directory, "rubiks-side-%s.png" % side_name)

        if os.path.exists(filename):
            rimg = RubiksImage(side_index, side_name)
            rimg.analyze_file(filename)
            data = merge_two_dicts(data, rimg.data)
        else:
            print "ERROR: %s does not exist" % filename
            sys.exit(1)

    print(json.dumps(data, sort_keys=True))
