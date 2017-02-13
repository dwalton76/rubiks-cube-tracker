#!/usr/bin/env python2

from rubikssquareextractor import RubiksImage
import argparse
import json
import logging
import sys


def merge_two_dicts(x, y):
    """
    Given two dicts, merge them into a new dict as a shallow copy.
    """
    z = x.copy()
    z.update(y)
    return z


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)12s %(levelname)8s: %(message)s')
log = logging.getLogger(__name__)

# Color the errors and warnings in red
logging.addLevelName(logging.ERROR, "\033[91m   %s\033[0m" % logging.getLevelName(logging.ERROR))
logging.addLevelName(logging.WARNING, "\033[91m %s\033[0m" % logging.getLevelName(logging.WARNING))


if len(sys.argv) > 1:
    filename = "/tmp/rubiks-side-%s.png" % sys.argv[1]
    rimg = RubiksImage(filename, debug=True)
    rimg.analyze()
    print(json.dumps(rimg.data, sort_keys=True))
else:
    data = {}

    for (side_index, side_name) in enumerate(('U', 'L', 'F', 'R', 'B', 'D')):
        filename = "/tmp/rubiks-side-%s.png" % side_name
        rimg = RubiksImage(filename, side_index, side_name)
        rimg.analyze()
        data = merge_two_dicts(data, rimg.data)

    print(json.dumps(data, sort_keys=True))
