#!/usr/bin/env python2

"""
Given the following .png files in /tmp/

jdoe@laptop ~/l/lego-crane-cuber> ls -l /tmp/rubiks-side-*
-rw-r--r-- 1 jdoe jdoe 105127 Jan 15 00:30 /tmp/rubiks-side-B.png
-rw-r--r-- 1 jdoe jdoe 105014 Jan 15 00:30 /tmp/rubiks-side-D.png
-rw-r--r-- 1 jdoe jdoe 103713 Jan 15 00:30 /tmp/rubiks-side-F.png
-rw-r--r-- 1 jdoe jdoe  99467 Jan 15 00:30 /tmp/rubiks-side-L.png
-rw-r--r-- 1 jdoe jdoe  98052 Jan 15 00:30 /tmp/rubiks-side-R.png
-rw-r--r-- 1 jdoe jdoe  97292 Jan 15 00:30 /tmp/rubiks-side-U.png
jdoe@laptop ~/l/lego-crane-cuber>

For each png
- find all of the rubiks squares
- json dump a dictionary that contains the RGB values for each square

"""

from copy import deepcopy
from pprint import pformat
import argparse
import cv2
import logging
import json
import logging
import math
import numpy as np
import os
import sys

# If you need to troubleshoot a particular image (say side F) run "./extract_rgb_pixels.py F"
# debug will be set to True in this scenario
debug = False

median_square_area = None
square_vs_non_square_debug_printed = []

def is_square(cX, cY, contour_area, approx, target_bounding_area=None, strict=False):
    """
    A few rules for a rubiks cube square
    - it has to have four sides
    - it must not be rotated (ie not a diamond)
    - (optional) it must be roughly the same size all of the other squares
    """
    global square_vs_non_square_debug_printed

    if (strict and len(approx) == 4) or (not strict and len(approx) >= 4):
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        bounding_area = float(w * h)
        contour_area_vs_bounding_area_ratio = float(contour_area / bounding_area)

        if strict:
            aspect_ratio_min = 0.70
            aspect_ratio_max = 1.30

            if contour_area_vs_bounding_area_ratio < 0.87:
                if (cX, cY) not in square_vs_non_square_debug_printed:
                    log.info("NOT SQUARE: (%d, %d) strict %s, contour_area_vs_bounding_area_ratio %s is less than 0.87" % (cX, cY, strict, contour_area_vs_bounding_area_ratio))
                    square_vs_non_square_debug_printed.append((cX, cY))
                return False

        else:
            aspect_ratio_min = 0.40
            aspect_ratio_max = 1.60

        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        if aspect_ratio >= aspect_ratio_min and aspect_ratio <= aspect_ratio_max:

            if target_bounding_area:
                bounding_area_ratio = float(target_bounding_area / bounding_area)

                if bounding_area_ratio >= 0.55 and bounding_area_ratio <= 1.35:
                    if (cX, cY) not in square_vs_non_square_debug_printed:
                        log.info("SQUARE: (%d, %d) strict %s, aspect_ratio %s, bounding_area_ratio %s, contour_area_vs_bounding_area_ratio %s" % (cX, cY, strict, aspect_ratio, bounding_area_ratio, contour_area_vs_bounding_area_ratio))
                        square_vs_non_square_debug_printed.append((cX, cY))
                    return True
                else:
                    if (cX, cY) not in square_vs_non_square_debug_printed:
                        log.info("NOT SQUARE: (%d, %d) strict %s, bounding_area %s, target_bounding_area %s, bounding_area_ratio %s" % (cX, cY, strict, bounding_area, target_bounding_area, bounding_area_ratio))
                        square_vs_non_square_debug_printed.append((cX, cY))
                    return False
            else:
                if (cX, cY) not in square_vs_non_square_debug_printed:
                    log.info("SQUARE: (%d, %d) strict %s, aspect_ratio %s, no target area, contour_area_vs_bounding_area_ratio %s" % (cX, cY, strict, aspect_ratio, contour_area_vs_bounding_area_ratio))
                    square_vs_non_square_debug_printed.append((cX, cY))
                return True
        else:
            if (cX, cY) not in square_vs_non_square_debug_printed:
                log.info("NOT SQUARE: (%d, %d) strict %s, aspect_ratio %s" % (cX, cY, strict, aspect_ratio))
                square_vs_non_square_debug_printed.append((cX, cY))
                return False

    #if (cX, cY) not in square_vs_non_square_debug_printed:
    #    log.info("NOT SQUARE: (%d, %d) strict %s, len approx %d" % (cX, cY, strict, len(approx)))
    #    square_vs_non_square_debug_printed.append((cX, cY))
    return False


def get_candidate_neighbors(target_tuple, candidates, img_width, img_height):
    """
    target_tuple is a contour, return stats on how many other contours are in
    the same 'row' or 'col' as target_tuple

    ROW_THRESHOLD determines how far up/down we look for other contours
    COL_THRESHOLD determines how far left/right we look for other contours
    """
    row_neighbors = 0
    row_square_neighbors = 0
    col_neighbors = 0
    col_square_neighbors = 0

    # These are percentages of the image width and height
    ROW_THRESHOLD = 0.03
    COL_THRESHOLD = 0.04

    width_wiggle = int(img_width * COL_THRESHOLD)
    height_wiggle = int(img_height * ROW_THRESHOLD)

    (_, _, _, _, target_cX, target_cY) = target_tuple

    log.debug("get_candidate_neighbors() for contour (%d, %d), width_wiggle %s, height_wiggle %s" %
        (target_cX, target_cY, width_wiggle, height_wiggle))

    for x in candidates:

        # do not count yourself
        if x == target_tuple:
            continue

        (index, area, currentContour, approx, cX, cY) = x
        x_delta = abs(cX - target_cX)
        y_delta = abs(cY - target_cY)

        if x_delta <= width_wiggle:
            col_neighbors += 1

            if is_square(cX, cY, area, approx, median_square_area):
                col_square_neighbors += 1
                log.debug("(%d, %d) is a square col neighbor" % (cX, cY))
            else:
                log.debug("(%d, %d) is a col neighbor but has %d corners" % (cX, cY, len(approx)))
        else:
            log.debug("(%d, %d) x delta %s it outside wiggle room %s" % (cX, cY, x_delta, width_wiggle))

        if y_delta <= height_wiggle:
            row_neighbors += 1

            if is_square(cX, cY, area, approx, median_square_area):
                row_square_neighbors += 1
                log.debug("(%d, %d) is a square row neighbor" % (cX, cY))
            else:
                log.debug("(%d, %d) is a row neighbor but has %d corners" % (cX, cY, len(approx)))
        else:
            log.debug("(%d, %d) y delta %s it outside wiggle room %s" % (cX, cY, y_delta, height_wiggle))

    log.debug("get_candidate_neighbors() for contour (%d, %d) has row %d, row_square %d, col %d, col_square %d neighbors" %
        (target_cX, target_cY, row_neighbors, row_square_neighbors, col_neighbors, col_square_neighbors))

    return (row_neighbors, row_square_neighbors, col_neighbors, col_square_neighbors)


def sort_by_row_col(candidates):
    """
    Given a set of candidates sort them starting from the upper left corner
    and ending at the bottom right corner
    """
    result = []
    num_squares = len(candidates)
    squares_per_row = int(math.sqrt(num_squares))

    for row_index in xrange(squares_per_row):

        # We want the squares_per_row that are closest to the top
        tmp = []
        for (index, area, currentContour, approx, cX, cY) in candidates:
            tmp.append((cY, cX))
        top_row = sorted(tmp)[:squares_per_row]

        # Now that we have those, sort them from left to right
        top_row_left_right = []
        for (cY, cX) in top_row:
            top_row_left_right.append((cX, cY))
        top_row_left_right = sorted(top_row_left_right)

        log.info("sort_by_row_col() row %d: %s" % (row_index, pformat(top_row_left_right)))
        candidates_to_remove = []
        for (target_cX, target_cY) in top_row_left_right:
            for (index, area, currentContour, approx, cX, cY) in candidates:
                if cX == target_cX and cY == target_cY:
                    result.append((index, area, currentContour, approx, cX, cY))
                    candidates_to_remove.append((index, area, currentContour, approx, cX, cY))
                    break

        for x in candidates_to_remove:
            candidates.remove(x)

    return result


def square_root_is_integer(integer):
    """
    Return True if integer's square root is an integer
    """
    root = math.sqrt(integer)

    if int(root + 0.5) ** 2 == integer:
        return True
    else:
        return False


def remove_contours_outside_cube(candidates, top, right, bottom, left):
    removed = 0
    candidates_to_remove = []

    # These are percentages of the image width and height
    ROW_THRESHOLD = 0.12
    COL_THRESHOLD = 0.10

    top = top * (1 - ROW_THRESHOLD)
    bottom = bottom * (1 + ROW_THRESHOLD)
    left = left * (1 - COL_THRESHOLD)
    right = right * (1 + COL_THRESHOLD)

    for x in candidates:
        (index, area, currentContour, approx, cX, cY) = x

        if cY < top or cY > bottom or cX < left or cX > right:
            candidates_to_remove.append(x)

    if candidates_to_remove:
        for x in candidates_to_remove:
            candidates.remove(x)
            removed += 1

    log.info("remove_contours_outside_cube() %d removed, %d remain" % (removed, len(candidates)))


def remove_rogue_non_squares(size, candidates, img_width, img_height):
    removed = 0
    candidates_to_remove = []

    for x in candidates:
        (index, area, currentContour, approx, cX, cY) = x

        if not is_square(cX, cY, area, approx, median_square_area):
            (row_neighbors, row_square_neighbors, col_neighbors, col_square_neighbors) =\
                get_candidate_neighbors(x, candidates, img_width, img_height)

            # not row_square_neighbors or not col_square_neighbors):
            if row_square_neighbors == size or col_square_neighbors == size:
                candidates_to_remove.append(x)

    if candidates_to_remove:
        for x in candidates_to_remove:
            candidates.remove(x)
            removed += 1

    log.info("remove_rogue_non_squares() %d removed, %d remain" % (removed, len(candidates)))


def remove_lonesome_contours(candidates, img_width, img_height, min_neighbors):
    """
    If a contour has less than min_neighbors in its row or col then remove
    this contour from candidates. We will also remove a contour if it does
    not have any square neighbors in its row or col as these are false
    positives.
    """
    log.info("remove_lonesome_contours() with less than %d neighbors" % min_neighbors)
    removed = 0

    while True:
        candidates_to_remove = []

        for x in candidates:
            (index, area, currentContour, approx, cX, cY) = x

            (row_neighbors, row_square_neighbors, col_neighbors, col_square_neighbors) =\
                get_candidate_neighbors(x, candidates, img_width, img_height)

            if row_neighbors < min_neighbors:
                candidates_to_remove.append(x)
                log.info("remove_lonesome_contours() (%d, %d) removed due to row_neighbors %d < %d" % (cX, cY, row_neighbors, min_neighbors))

            elif col_neighbors < min_neighbors:
                candidates_to_remove.append(x)
                log.info("remove_lonesome_contours() (%d, %d) removed due to col_neighbors %d < %d" % (cX, cY, col_neighbors, min_neighbors))

            elif not row_square_neighbors and not col_square_neighbors:
                candidates_to_remove.append(x)
                log.info("remove_lonesome_contours() (%d, %d) removed due to no row and col square neighbors" % (cX, cY))

        if candidates_to_remove:
            for x in candidates_to_remove:
                candidates.remove(x)
                removed += 1
        else:
            break

    log.info("remove_lonesome_contours() %d removed, %d remain" % (removed, len(candidates)))


def get_cube_size(candidates, img_width, img_height):
    """
    Look at all of the contours that are squares and see how many square
    neighbors they have in their row and col. Store the number of square
    contours in each row/col in data, then sort data and return the
    median entry
    """
    # Find the median area of all square contours and find the top, right,
    # bottom, left boundry of all square contours
    square_areas = []
    for x in candidates:
        (index, area, currentContour, approx, cX, cY) = x

        if is_square(cX, cY, area, approx, strict=True):
            square_areas.append(int(area))

    global square_vs_non_square_debug_printed
    square_vs_non_square_debug_printed = []
    log.warning("reset SQUARE vs NOT SQUARE debugs")
    square_areas = sorted(square_areas)
    num_squares = len(square_areas)
    median_square_area_index = int(num_squares/2)
    global median_square_area
    median_square_area = int(square_areas[median_square_area_index])

    log.info("%d squares, median index %d, median area %d, all square areas %s" %\
        (num_squares, median_square_area_index, median_square_area, ','.join(map(str, square_areas))))

    # Now find all of the square contours that are the same size (roughly) as the
    # median square size. Look to see how many square neighbors are in the row and
    # col for each square contour.
    data = []
    top = None
    right = None
    bottom = None
    left = None

    for x in candidates:
        (index, area, currentContour, approx, cX, cY) = x

        if is_square(cX, cY, area, approx, median_square_area, strict=True):
            (row_neighbors, row_square_neighbors, col_neighbors, col_square_neighbors) =\
                get_candidate_neighbors(x, candidates, img_width, img_height)
            row_size = row_square_neighbors + 1
            col_size = col_square_neighbors + 1

            if row_size > col_size:
                data.append(row_size)
            else:
                data.append(col_size)

            if top is None or cY < top:
                top = cY

            if bottom is None or cY > bottom:
                bottom = cY

            if left is None or cX < left:
                left = cX

            if right is None or cX > right:
                right = cX

    data = sorted(data)
    median_index = int(len(data)/2)
    median_size = data[median_index]

    log.info("cube size...%d squares, data %s" % (len(data), ','.join(map(str, data))))
    log.warning("cube size is %d, top %s, right %s, bottom %s, left %s" %\
        (median_size, top, right, bottom, left))

    return (median_size, top, right, bottom, left)


def draw_cube(image, candidates, desc):

    if not debug:
        return

    if candidates:
        to_draw = []
        to_draw_square = []
        to_draw_approx = []

        for (index, area, contour, approx, cX, cY) in candidates:
            if is_square(cX, cY, area, approx, median_square_area):
                to_draw_square.append(contour)
                # to_draw_approx.append(approx)
            else:
                to_draw.append(contour)
                to_draw_approx.append(approx)

        tmp_image = image.copy()
        cv2.drawContours(tmp_image, to_draw, -1, (255, 0, 0), 2)
        cv2.drawContours(tmp_image, to_draw_approx, -1, (0, 255, 0), 2)
        cv2.drawContours(tmp_image, to_draw_square, -1, (0, 0, 255), 2)

        cv2.imshow(desc, tmp_image)
        cv2.waitKey(0)
    else:
        cv2.imshow(desc, image)
        cv2.waitKey(0)


def get_rubiks_squares(filename):
    image = cv2.imread(filename)
    (img_height, img_width) = image.shape[:2]

    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #draw_cube(gray, None, "gray")

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    #draw_cube(blurred, None, "blurred")

    # Threshold settings from here:
    # http://opencvpython.blogspot.com/2012/06/sudoku-solver-part-2.html
    thresh = cv2.adaptiveThreshold(blurred, 255, 1, 1, 11, 2)
    #draw_cube(thresh, None, "thresh")

    # Use a very high h value so that we really blur the image to remove
    # all spots that might be in the rubiks squares...we want the rubiks
    # squares to be solid black
    denoised = cv2.fastNlMeansDenoising(thresh, h=50)
    draw_cube(denoised, None, "denoised")

    # Now invert the image so that the rubiks squares are white but most
    # of the rest of the image is black
    inverted = cv2.threshold(denoised, 10, 255, cv2.THRESH_BINARY_INV)[1]
    #draw_cube(inverted, None, "inverted")

    # Erode the image to remove any really thin lines that might be connected our squares
    kernel = np.ones((4,4), np.uint8)
    eroded = cv2.erode(inverted, kernel, iterations=2)
    #draw_cube(eroded, None, "eroded")

    # Now dilate the image to make the squares a little larger
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    draw_cube(dilated, None, "dilated")

    (contours, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    hierarchy = hierarchy[0] # get the actual inner list of hierarchy descriptions

    # http://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html
    # http://stackoverflow.com/questions/11782147/python-opencv-contour-tree-hierarchy
    #
    # For each contour, find the bounding rectangle and draw it
    index = 0
    for component in zip(contours, hierarchy):
        currentContour = component[0]
        currentHierarchy = component[1]

        '''
        Things I used to filter on here but no longer do
        - currentHierarchy[2] of -1 means this contour has no children so we know
          this is the "inside" contour for a square...some squares get two contours
          due to the black border around the edge of the square

          This ended up filtering out some legit contours of squares so I chopped it

        - using 'approx' to determine if the contour has as least 4 corners

          If the square has dent in it (or a splotch, dust, etc) this can cause us to
          find a contour inside the square but the contour won't be square at all
        '''
        # approximate the contour
        peri = cv2.arcLength(currentContour, True)
        approx = cv2.approxPolyDP(currentContour, 0.1 * peri, True)
        area = cv2.contourArea(currentContour)

        if currentHierarchy[3] == -1:
            has_parent = False
        else:
            has_parent = True

        # Sometimes dents in the sticker on a square can cause us to find a contour
        # within the contour for the square.  Ignore any contour that has a parent contour.
        if has_parent:
            continue

        if area > 30:

            # compute the center of the contour
            M = cv2.moments(currentContour)

            if M["m00"]:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                log.info("(%d, %d), area %d, corners %d" % (cX, cY, area, len(approx)))
                candidates.append((index, area, currentContour, approx, cX, cY))
        index += 1

    draw_cube(image, candidates, "pre lonesome removal #1")
    remove_lonesome_contours(candidates, img_width, img_height, 1)
    draw_cube(image, candidates, "post lonesome removal #1")

    # get the extreme coordinates for any of the obvious squares and
    # remove all contours outside of those coordinates
    (size, top, right, bottom, left) =\
        get_cube_size(deepcopy(candidates), img_width, img_height)
    remove_contours_outside_cube(candidates, top, right, bottom, left)
    draw_cube(image, candidates, "post outside square removal")

    remove_rogue_non_squares(size, candidates, img_width, img_height)
    draw_cube(image, candidates, "post rogue non-square removal")

    num_squares = len(candidates)

    if not square_root_is_integer(num_squares):
        remove_lonesome_contours(candidates, img_width, img_height, int(size/2))
        draw_cube(image, candidates, "post lonesome removal #2")
        num_squares = len(candidates)

    candidates = sort_by_row_col(deepcopy(candidates))
    data = []

    for (index, area, contour, approx, cX, cY) in candidates:
        # We used to use the value of the center pixel
        #(blue, green, red) = map(int, image[cY, cX])
        #data.append((red, green, blue))

        # Now we use the mean value of the contour
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        (mean_blue, mean_green, mean_red, _)= map(int, cv2.mean(image, mask = mask))
        data.append((mean_red, mean_green, mean_blue))

        #log.info("normal BGR (%s, %s, %s), mean BGR (%s, %s, %s)" %\
        #    (blue, green, red, mean_blue, mean_green, mean_red))

    draw_cube(image, candidates, "Final")

    # Verify we found the right number of squares
    num_squares = len(candidates)

    if not square_root_is_integer(num_squares):
        raise Exception("Found %d squares which cannot be right" % num_squares)

    return data


def rotate_2d_array(original):
    """
    http://stackoverflow.com/questions/8421337/rotating-a-two-dimensional-array-in-python
    """
    result = []
    for x in zip(*original[::-1]):
        result.append(x)
    return result


def compress_2d_array(original):
    """
    Convert 2d array to a 1d array
    """
    result = []
    for row in original:
        for col in row:
            result.append(col)
    return result


def extract_rgb_pixels(target_side):
    global debug
    global square_vs_non_square_debug_printed
    global median_square_area

    colors = {}
    prev_squares_per_side = None
    prev_side = None
    squares_per_side = None

    for (side_index, side) in enumerate(('U', 'L', 'F', 'R', 'B', 'D')):
        '''
        squares are numbered like so:

               01 02
               03 04
        05 06  09 10  13 14  17 18
        07 08  11 12  15 16  19 20
               21 22
               23 24

        calculate the index of the first square for each side
        '''

        # target_side is only non-None when we are debugging the image for a specific side
        if target_side is not None:
            if side == target_side:
                debug = True
            else:
                debug = False
                continue

        filename = "/tmp/rubiks-side-%s.png" % side
        log.warning("Analyze %s" % filename)
        median_square_area = None

        if not os.path.exists(filename):
            print "ERROR: %s does not exists" % filename
            sys.exit(1)

        # data will be a list of (R, G, B) tuples, one entry for each square on a side
        data = get_rubiks_squares(filename)
        log.info("squares RGB data\n%s\n" % pformat(data))

        squares_per_side = len(data)
        size = int(math.sqrt(squares_per_side))
        init_square_index = (side_index * squares_per_side) + 1

        if prev_squares_per_side is not None:
            assert squares_per_side == prev_squares_per_side,\
                "side %s had %d squares, side %s has %d squares" % (prev_side, prev_squares_per_side, side, squares_per_side)

        square_indexes = []
        for row in range(size):
            square_indexes_for_row = []
            for col in range(size):
                square_indexes_for_row.append(init_square_index + (row * size) + col)
            square_indexes.append(square_indexes_for_row)

        '''
        The L, F, R, and B sides are simple, for the U and D sides the cube in
        the png is rotated by 90 degrees so we need to rotate our array of
        square indexes by 90 degrees to compensate
        '''
        if side == 'U' or side == 'D':
            my_indexes = rotate_2d_array(square_indexes)
        else:
            my_indexes = square_indexes

        log.info("%s square_indexes\n%s\n" % (side, pformat(square_indexes)))
        log.info("%s my_indexes\n%s\n" % (side, pformat(my_indexes)))
        my_indexes = compress_2d_array(my_indexes)
        log.info("%s my_indexes (final) %s" % (side, str(my_indexes)))

        for index in range(squares_per_side):
            square_index = my_indexes[index]
            (red, green, blue) = data[index]
            log.info("square %d RGB (%d, %d, %d)" % (square_index, red, green, blue))

            # colors is a dict where the square number (as an int) will be
            # the key and a RGB tuple the value
            colors[square_index] = (red, green, blue)

        prev_squares_per_side = squares_per_side
        prev_side = side
        log.info("\n\n\n")

    return colors


if __name__ == '__main__':
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
