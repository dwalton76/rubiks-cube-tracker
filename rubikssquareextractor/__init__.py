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

log = logging.getLogger(__name__)

# If you need to troubleshoot a particular image (say side F) run "rubiks-square-extractor.py F"
# debug will be set to True in this scenario
debug = False

median_square_area = None
square_vs_non_square_debug_printed = []
contours_by_index = {}


def pixel_distance(A, B):
    """
    In 9th grade I sat in geometry class wondering "when then hell am I
    ever going to use this?"...today is that day.

    Return the distance between two pixels
    """
    (col_A, row_A) = A
    (col_B, row_B) = B

    return math.sqrt(math.pow(col_B - col_A, 2) + math.pow(row_B - row_A, 2))


def get_angle(A, B, C, debug=False):
    """
    Return the angle at C (in radians) for the triangle formed by A, B, C
    a, b, c are lengths

        C
       / \
    b /   \a
     /     \
    A-------B
        c

    """
    (col_A, row_A) = A
    (col_B, row_B) = B
    (col_C, row_C) = C
    a = pixel_distance(C, B)
    b = pixel_distance(A, C)
    c = pixel_distance(A, B)

    try:
        cos_angle = (math.pow(a, 2) + math.pow(b, 2) - math.pow(c, 2)) / (2 * a * b)
    except ZeroDivisionError as e:
        log.warning("get_angle: A %s, B %s, C %s, a %.3f, b %.3f, c %.3f" % (A, B, C, a, b, c))
        raise e

    # If CA and CB are very long and the angle at C very narrow we can get an
    # invalid cos_angle which will cause math.acos() to raise a ValueError exception
    if cos_angle > 1:
        cos_angle = 1
    elif cos_angle < -1:
        cos_angle = -1

    angle_ACB = math.acos(cos_angle)
    if debug:
        log.info("get_angle: A %s, B %s, C %s, a %.3f, b %.3f, c %.3f, cos_angle %s, angle_ACB %s" %
                 (A, B, C, a, b, c, pformat(cos_angle), int(math.degrees(angle_ACB))))
    return angle_ACB


def sort_corners(corner1, corner2, corner3, corner4):
    """
    Sort the corners such that
    - A is top left
    - B is top right
    - C is bottom left
    - D is bottom right

    Return an (A, B, C, D) tuple
    """
    results = []
    corners = (corner1, corner2, corner3, corner4)

    min_x = None
    max_x = None
    min_y = None
    max_y = None

    for (x, y) in corners:
        if min_x is None or x < min_x:
            min_x = x

        if max_x is None or x > max_x:
            max_x = x

        if min_y is None or y < min_y:
            min_y = y

        if max_y is None or y > max_y:
            max_y = y

    # top left
    top_left = None
    top_left_distance = None
    for (x, y) in corners:
        distance = pixel_distance((min_x, min_y), (x, y))
        if top_left_distance is None or distance < top_left_distance:
            top_left = (x, y)
            top_left_distance = distance

    results.append(top_left)

    # top right
    top_right = None
    top_right_distance = None

    for (x, y) in corners:
        if (x, y) in results:
            continue

        distance = pixel_distance((max_x, min_y), (x, y))
        if top_right_distance is None or distance < top_right_distance:
            top_right = (x, y)
            top_right_distance = distance
    results.append(top_right)

    # bottom left
    bottom_left = None
    bottom_left_distance = None

    for (x, y) in corners:
        if (x, y) in results:
            continue

        distance = pixel_distance((min_x, max_y), (x, y))

        if bottom_left_distance is None or distance < bottom_left_distance:
            bottom_left = (x, y)
            bottom_left_distance = distance
    results.append(bottom_left)

    # bottom right
    bottom_right = None
    bottom_right_distance = None

    for (x, y) in corners:
        if (x, y) in results:
            continue

        distance = pixel_distance((max_x, max_y), (x, y))

        if bottom_right_distance is None or distance < bottom_right_distance:
            bottom_right = (x, y)
            bottom_right_distance = distance
    results.append(bottom_right)

    return results


def approx_is_square(approx):
    """
    Rules:
    - there must be four corners
    - all four lines must be roughly the same length
    - all four corners must be roughly 90 degrees
    """

    # there must be four corners
    if len(approx) != 4:
        return False

    (A, B, C, D) = sort_corners(tuple(approx[0][0]),
                                tuple(approx[1][0]),
                                tuple(approx[2][0]),
                                tuple(approx[3][0]))

    # all four lines must be roughly the same length
    AB = pixel_distance(A, B)
    BC = pixel_distance(B, C)
    CD = pixel_distance(C, D)
    DA = pixel_distance(D, A)
    distances = (AB, BC, CD, DA)
    max_distance = max(distances)
    cutoff = int(max_distance/2)

    # log.info("approx_is_square A %s, B, %s, C %s, D %s, distance AB %d, BC %d, CD %d, DA %d, max %d, cutoff %d" %
    #          (A, B, C, D, AB, BC, CD, DA, max_distance, cutoff))

    # If any of the sides is less than half the length of the
    # longest side then this is obviously not a square
    for distance in distances:
        if abs(max_distance - distance) > cutoff:
            return False

    # all four corners must be roughly 90 degrees
    angle_cutoff = 20
    min_angle = 90 - angle_cutoff
    max_angle = 90 + angle_cutoff

    # Angle at A
    angle_A = int(math.degrees(get_angle(C, B, A)))
    if angle_A < min_angle or angle_A > max_angle:
        return False

    # Angle at B
    angle_B = int(math.degrees(get_angle(A, D, B)))
    if angle_B < min_angle or angle_B > max_angle:
        return False

    # Angle at C
    angle_C = int(math.degrees(get_angle(A, D, C)))
    if angle_C < min_angle or angle_C > max_angle:
        return False

    # Angle at D
    angle_D = int(math.degrees(get_angle(C, B, D)))
    if angle_D < min_angle or angle_D > max_angle:
        return False

    return True


class CustomContour(object):

    def __init__(self, index, contour, heirarchy):
        self.index = index
        self.contour = contour
        self.heirarchy = heirarchy
        peri = cv2.arcLength(contour, True)
        self.approx = cv2.approxPolyDP(contour, 0.1 * peri, True)
        self.area = cv2.contourArea(contour)
        self.corners = len(self.approx)

        if self.heirarchy[3] == -1:
            self.has_parent = False
        else:
            self.has_parent = True

        # compute the center of the contour
        M = cv2.moments(contour)

        if M["m00"]:
            self.cX = int(M["m10"] / M["m00"])
            self.cY = int(M["m01"] / M["m00"])
            # log.info("(%d, %d), area %d, corners %d" % (self.cX, self.cY, self.area, self.corners))
        else:
            self.cX = None
            self.cY = None

    def __str__(self):
        return "Contour #%d (%s, %s)" % (self.index, self.cX, self.cY)

    def is_square(self, target_area=None, strict=False):
        """
        A few rules for a rubiks cube square
        - it has to have four sides
        - it must not be rotated (ie not a diamond)
        - (optional) it must be roughly the same size all of the other squares
        """
        global square_vs_non_square_debug_printed

        cX = self.cX
        cY = self.cY
        contour_area = self.area
        approx = self.approx

        if not approx_is_square(approx):
            return False

        if target_area:
            area_ratio = float(target_area / self.area)

            if area_ratio < 0.80 or area_ratio > 1.20:
                # log.info("FALSE %s target_area %d, my area %d, ratio %s" % (self, target_area, self.area, area_ratio))
                return False
            else:
                # log.info("TRUE %s target_area %d, my area %d, ratio %s" % (self, target_area, self.area, area_ratio))
                return True
        else:
            return True

        # dwalton - experiment...ignore this for now
        '''
        if (strict and self.corners == 4) or (not strict and self.corners >= 4):
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            bounding_area = float(w * h)

            if strict:
                aspect_ratio_min = 0.70
                aspect_ratio_max = 1.30
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
                            log.info("SQUARE: %s strict %s, aspect_ratio %s, bounding_area_ratio %s" %
                                     (self, strict, aspect_ratio, bounding_area_ratio))
                            square_vs_non_square_debug_printed.append((cX, cY))
                        return True
                    else:
                        if (cX, cY) not in square_vs_non_square_debug_printed:
                            log.info("NOT SQUARE: %s strict %s, bounding_area %s, target_bounding_area %s, bounding_area_ratio %s" %
                                     (self, strict, bounding_area, target_bounding_area, bounding_area_ratio))
                            square_vs_non_square_debug_printed.append((cX, cY))
                        return False
                else:
                    if (cX, cY) not in square_vs_non_square_debug_printed:
                        log.info("SQUARE: %s strict %s, aspect_ratio %s, no target area" %
                                 (self, strict, aspect_ratio))
                        square_vs_non_square_debug_printed.append((cX, cY))
                    return True
            else:
                if (cX, cY) not in square_vs_non_square_debug_printed:
                    log.info("NOT SQUARE: %s strict %s, aspect_ratio %s" % (self, strict, aspect_ratio))
                    square_vs_non_square_debug_printed.append((cX, cY))
                    return False

        return False
        '''

    def get_child(self):
        # Each contour has its own information regarding what hierarchy it
        # is, who is its parent, who is its parent etc. OpenCV represents it as
        # an array of four values : [Next, Previous, First_parent, Parent]
        child = self.heirarchy[2]

        if child != -1:
            return contours_by_index[child]
        return None

    def child_is_square(self):
        """
        The black border between the squares can cause us to sometimes find a
        contour for the outside edge of the border and a contour for the the
        inside edge.  This function returns True if this contour is the outside
        contour in that scenario.
        """
        child_con = self.get_child()

        if child_con:
            # If there is a dent in a square sometimes you will get a really small
            # contour inside the square...we want to ignore those so make sure the
            # area of the inner square is close to the area of the outer square.
            if int(child_con.area * 2) < self.area:
                return False

            if child_con.is_square(strict=True):
                return True

        return False

    def get_parent(self):
        # Each contour has its own information regarding what hierarchy it
        # is, who is its parent, who is its parent etc. OpenCV represents it as
        # an array of four values : [Next, Previous, First_parent, Parent]
        parent = self.heirarchy[3]

        if parent != -1:
            return contours_by_index[parent]
        return None

    def parent_is_square(self):
        parent_con = self.get_parent()

        if parent_con:
            # If there is a dent in a square sometimes you will get a really small
            # contour inside the square...we want to ignore those so make sure the
            # area of the inner square is close to the area of the outer square.
            if int(parent_con.area * 2) < self.area:
                return False

            if parent_con.is_square(strict=True):
                return True

        return False


def get_candidate_neighbors(target_con, candidates, img_width, img_height):
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

    target_cX = target_con.cX
    target_cY = target_con.cY

    log.debug("get_candidate_neighbors() for contour (%d, %d), width_wiggle %s, height_wiggle %s" %
        (target_cX, target_cY, width_wiggle, height_wiggle))

    for con in candidates:

        # do not count yourself
        if con == target_con:
            continue

        x_delta = abs(con.cX - target_cX)
        y_delta = abs(con.cY - target_cY)

        if x_delta <= width_wiggle:
            col_neighbors += 1

            if con.is_square(median_square_area):
                col_square_neighbors += 1
                log.debug("(%d, %d) is a square col neighbor" % (con.cX, con.cY))
            else:
                log.debug("(%d, %d) is a col neighbor but has %d corners" % (con.cX, con.cY, con.corners))
        else:
            log.debug("(%d, %d) x delta %s it outside wiggle room %s" % (con.cX, con.cY, x_delta, width_wiggle))

        if y_delta <= height_wiggle:
            row_neighbors += 1

            if con.is_square(median_square_area):
                row_square_neighbors += 1
                log.debug("(%d, %d) is a square row neighbor" % (con.cX, con.cY))
            else:
                log.debug("(%d, %d) is a row neighbor but has %d corners" % (con.cX, con.cY, con.corners))
        else:
            log.debug("(%d, %d) y delta %s it outside wiggle room %s" % (con.cX, con.cY, y_delta, height_wiggle))

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
        for con in candidates:
            tmp.append((con.cY, con.cX))
        top_row = sorted(tmp)[:squares_per_row]

        # Now that we have those, sort them from left to right
        top_row_left_right = []
        for (cY, cX) in top_row:
            top_row_left_right.append((cX, cY))
        top_row_left_right = sorted(top_row_left_right)

        log.info("sort_by_row_col() row %d: %s" % (row_index, pformat(top_row_left_right)))
        candidates_to_remove = []
        for (target_cX, target_cY) in top_row_left_right:
            for con in candidates:

                if con in candidates_to_remove:
                    continue

                if con.cX == target_cX and con.cY == target_cY:
                    result.append(con)
                    candidates_to_remove.append(con)
                    break

        for con in candidates_to_remove:
            candidates.remove(con)

    assert len(result) == num_squares, "sort_by_row_col is returning %d squares, it should be %d" % (len(result), num_squares)
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

    for con in candidates:
        if con.cY < top or con.cY > bottom or con.cX < left or con.cX > right:
            candidates_to_remove.append(con)

    if candidates_to_remove:
        for con in candidates_to_remove:
            candidates.remove(con)
            removed += 1

    log.info("remove_contours_outside_cube() %d removed, %d remain" % (removed, len(candidates)))


def remove_rogue_non_squares(size, candidates, img_width, img_height):
    removed = 0
    candidates_to_remove = []

    for con in candidates:
        if not con.is_square(median_square_area):
            (row_neighbors, row_square_neighbors, col_neighbors, col_square_neighbors) =\
                get_candidate_neighbors(con, candidates, img_width, img_height)

            if not row_square_neighbors or not col_square_neighbors or row_square_neighbors == size or col_square_neighbors == size:
                candidates_to_remove.append(con)

    if candidates_to_remove:
        for con in candidates_to_remove:
            candidates.remove(con)
            removed += 1

    log.info("remove_rogue_non_squares() %d removed, %d remain" % (removed, len(candidates)))


def remove_square_within_square_contours(candidates):
    removed = 0
    candidates_to_remove = []

    # Remove parents with square child contours
    for con in candidates:
        if con.is_square(strict=True) and con.child_is_square():
            candidates_to_remove.append(con)

    # Remove contours whose parents are square
    for con in candidates:
        if con in candidates_to_remove:
            continue

        if con.parent_is_square():
            parent = con.get_parent()

            if parent not in candidates_to_remove:
                candidates_to_remove.append(con)

    if candidates_to_remove:
        for x in candidates_to_remove:
            candidates.remove(x)
            removed += 1


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

        for con in candidates:
            (row_neighbors, row_square_neighbors, col_neighbors, col_square_neighbors) =\
                get_candidate_neighbors(con, candidates, img_width, img_height)

            if row_neighbors < min_neighbors:
                candidates_to_remove.append(con)
                log.info("remove_lonesome_contours() %s removed due to row_neighbors %d < %d" % (con, row_neighbors, min_neighbors))

            elif col_neighbors < min_neighbors:
                candidates_to_remove.append(con)
                log.info("remove_lonesome_contours() %s removed due to col_neighbors %d < %d" % (con, col_neighbors, min_neighbors))

            elif not row_square_neighbors and not col_square_neighbors:
                candidates_to_remove.append(con)
                log.info("remove_lonesome_contours() %s removed due to no row and col square neighbors" % con)

            elif con.corners == 2:
                candidates_to_remove.append(con)
                log.info("remove_lonesome_contours() %s removed due only two corners" % con)

        if candidates_to_remove:
            for con in candidates_to_remove:
                candidates.remove(con)
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
    for con in candidates:
        if con.is_square(strict=True):
            square_areas.append(int(con.area))

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

    for con in candidates:
        if con.is_square(median_square_area, strict=True):
            (row_neighbors, row_square_neighbors, col_neighbors, col_square_neighbors) =\
                get_candidate_neighbors(con, candidates, img_width, img_height)
            row_size = row_square_neighbors + 1
            col_size = col_square_neighbors + 1

            if row_size > col_size:
                data.append(row_size)
            else:
                data.append(col_size)

            if top is None or con.cY < top:
                top = con.cY

            if bottom is None or con.cY > bottom:
                bottom = con.cY

            if left is None or con.cX < left:
                left = con.cX

            if right is None or con.cX > right:
                right = con.cX

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
    log.info("draw_cube() for %s" % desc)

    if candidates:
        to_draw = []
        to_draw_square = []
        to_draw_approx = []

        for con in candidates:
            if con.is_square(strict=True):
                to_draw_square.append(con.contour)
                #to_draw_approx.append(con.approx)
            else:
                to_draw.append(con.contour)
                to_draw_approx.append(con.approx)

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
    global contours_by_index

    image = cv2.imread(filename)
    (img_height, img_width) = image.shape[:2]

    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #draw_cube(gray, None, "00 gray")

    # blur a little...not sure why but most canny examples I've found do this prior to running canny
    # http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    #draw_cube(blurred, None, "10 blurred")

    # canny to find the edges
    canny = cv2.Canny(blurred, 20, 40)
    draw_cube(canny, None, "20 canny")

    # dilate the image to make the edge lines thicker
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(canny, kernel, iterations=1)
    draw_cube(dilated, None, "30 dilated")

    # Erode the image to remove any really thin lines that might be connected our squares
    #eroded = cv2.erode(dilated, kernel, iterations=1)
    #draw_cube(eroded, None, "35 eroded")

    # Now invert the image so that the rubiks squares are white but most
    # of the rest of the image is black
    #inverted = cv2.threshold(dilated, 10, 255, cv2.THRESH_BINARY_INV)[1]
    #draw_cube(inverted, None, "35 inverted")


    # find the contours and create a CustomContour object for each...store these in "candidates"
    # http://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html
    # http://stackoverflow.com/questions/11782147/python-opencv-contour-tree-hierarchy
    (contours, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # (contours, hierarchy) = cv2.findContours(inverted.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    hierarchy = hierarchy[0] # get the actual inner list of hierarchy descriptions

    index = 0
    for component in zip(contours, hierarchy):
        con = CustomContour(index, component[0], component[1])
        contours_by_index[index] = con

        if con.area > 30 and con.cX is not None:
            candidates.append(con)
        index += 1

    # dwalton
    draw_cube(image, candidates, "40 pre square-within-square removal #1")
    remove_square_within_square_contours(candidates)
    #draw_cube(image, candidates, "post square-within-square removal #1")

    draw_cube(image, candidates, "50 pre lonesome removal #1")
    remove_lonesome_contours(candidates, img_width, img_height, 1)
    #draw_cube(image, candidates, "post lonesome removal #1")

    # get the extreme coordinates for any of the obvious squares and
    # remove all contours outside of those coordinates
    draw_cube(image, candidates, "60 pre outside cube removal")
    (size, top, right, bottom, left) =\
        get_cube_size(deepcopy(candidates), img_width, img_height)
    remove_contours_outside_cube(candidates, top, right, bottom, left)

    draw_cube(image, candidates, "70 pre non-square removal")
    remove_rogue_non_squares(size, candidates, img_width, img_height)
    num_squares = len(candidates)

    if square_root_is_integer(num_squares):
        draw_cube(image, candidates, "80 post non-square removal")
    else:
        draw_cube(image, candidates, "80 pre lonesome removal #2")
        remove_lonesome_contours(candidates, img_width, img_height, int(size/2))
        num_squares = len(candidates)

    # dwalton
    draw_cube(image, candidates, "90 Final")

    # Verify we found a feasible number of squares
    num_squares = len(candidates)

    if square_root_is_integer(num_squares):
        log.info("Found %d squares" % num_squares)
    else:
        raise Exception("Found %d squares which cannot be right" % num_squares)

    # Verify each row/col has the same number of neighbors
    req_row_neighbors = None
    req_col_neighbors = None

    for con in candidates:
        (row_neighbors, row_square_neighbors, col_neighbors, col_square_neighbors) =\
            get_candidate_neighbors(con, candidates, img_width, img_height)

        if req_row_neighbors is None:
            req_row_neighbors = row_neighbors

        if req_col_neighbors is None:
            req_col_neighbors = col_neighbors

        assert row_neighbors == req_row_neighbors, "%s has %d row neighbors, must be %d" % (con, row_neighbors, req_row_neighbors)
        assert col_neighbors == req_col_neighbors, "%s has %d col neighbors, must be %d" % (con, col_neighbors, req_col_neighbors)

    data = []
    for con in sort_by_row_col(deepcopy(candidates)):
        # Use the mean value of the contour
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [con.contour], 0, 255, -1)
        (mean_blue, mean_green, mean_red, _)= map(int, cv2.mean(image, mask = mask))
        data.append((mean_red, mean_green, mean_blue))

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
