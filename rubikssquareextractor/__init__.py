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
from itertools import combinations
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

    The corners will be in order:

        A ---- B
        |      |
        |      |
        C ---- D
    """

    # There must be four corners
    if len(approx) != 4:
        return False

    # Find the four corners
    (A, B, C, D) = sort_corners(tuple(approx[0][0]),
                                tuple(approx[1][0]),
                                tuple(approx[2][0]),
                                tuple(approx[3][0]))

    # Find the lengths of all four sides
    AB = pixel_distance(A, B)
    AC = pixel_distance(A, C)
    DB = pixel_distance(D, B)
    DC = pixel_distance(D, C)
    distances = (AB, AC, DB, DC)
    max_distance = max(distances)
    cutoff = int(max_distance * 0.60)

    #log.info("approx_is_square A %s, B, %s, C %s, D %s, distance AB %d, AC %d, DB %d, DC %d, max %d, cutoff %d" %
    #         (A, B, C, D, AB, AC, DB, DC, max_distance, cutoff))

    # If any side is much smaller than the longest side, return False
    for distance in distances:
        if distance < cutoff:
            return False

    # all four corners must be roughly 90 degrees
    angle_threshold = 20
    min_angle = 90 - angle_threshold
    max_angle = 90 + angle_threshold

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


def square_root_is_integer(integer):
    """
    Return True if integer's square root is an integer
    """
    root = math.sqrt(integer)

    if int(root + 0.5) ** 2 == integer:
        return True
    else:
        return False


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


class CustomContour(object):

    def __init__(self, rubiks_parent, index, contour, heirarchy):
        self.rubiks_parent = rubiks_parent
        self.index = index
        self.contour = contour
        self.heirarchy = heirarchy
        peri = cv2.arcLength(contour, True)
        self.approx = cv2.approxPolyDP(contour, 0.1 * peri, True)
        self.area = cv2.contourArea(contour)
        self.corners = len(self.approx)

        # compute the center of the contour
        M = cv2.moments(contour)

        if M["m00"]:
            self.cX = int(M["m10"] / M["m00"])
            self.cY = int(M["m01"] / M["m00"])
        else:
            self.cX = None
            self.cY = None

    def __str__(self):
        return "Contour #%d (%s, %s)" % (self.index, self.cX, self.cY)

    def is_square(self, target_area=None):

        if not approx_is_square(self.approx):
            return False

        if target_area:
            area_ratio = float(target_area / self.area)

            if area_ratio < 0.75 or area_ratio > 1.25:
                # log.info("FALSE %s target_area %d, my area %d, ratio %s" % (self, target_area, self.area, area_ratio))
                return False
            else:
                # log.info("TRUE %s target_area %d, my area %d, ratio %s" % (self, target_area, self.area, area_ratio))
                return True
        else:
            return True

    def get_child(self):
        # Each contour has its own information regarding what hierarchy it
        # is, who is its parent, who is its parent etc. OpenCV represents it as
        # an array of four values : [Next, Previous, First_parent, Parent]
        child = self.heirarchy[2]

        if child != -1:
            return self.rubiks_parent.contours_by_index[child]
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

            if child_con.is_square():
                return True

        return False

    def get_parent(self):
        # Each contour has its own information regarding what hierarchy it
        # is, who is its parent, who is its parent etc. OpenCV represents it as
        # an array of four values : [Next, Previous, First_parent, Parent]
        parent = self.heirarchy[3]

        if parent != -1:
            return self.rubiks_parent.contours_by_index[parent]
        return None

    def parent_is_square(self):
        parent_con = self.get_parent()

        if parent_con:
            # If there is a dent in a square sometimes you will get a really small
            # contour inside the square...we want to ignore those so make sure the
            # area of the inner square is close to the area of the outer square.
            if int(parent_con.area * 2) < self.area:
                return False

            if parent_con.is_square():
                return True

        return False


class RubiksImage(object):

    def __init__(self, filename, index=0, name='foo', debug=False):
        self.filename = filename
        self.index = index
        self.name = name
        self.data = {}
        self.candidates = []
        self.contours_by_index = {}
        self.debug = debug
        self.img_height = None
        self.img_width = None
        self.size = None
        self.square_area = None
        self.top = None
        self.right = None
        self.bottom = None
        self.left = None

        if not os.path.exists(filename):
            print "ERROR: %s does not exists" % filename
            sys.exit(1)

    def __str__(self):
        return self.name

    def draw_cube(self, image, desc, missing=[]):

        if not self.debug:
            return

        log.info("draw_cube() for %s" % desc)

        if self.candidates:
            to_draw = []
            to_draw_square = []
            to_draw_approx = []
            to_draw_missing = []
            to_draw_missing_approx = []

            for con in missing:
                to_draw_missing.append(con.contour)
                to_draw_missing_approx.append(con.approx)

            for con in self.candidates:
                if con in missing:
                    continue

                if con.is_square():
                    to_draw_square.append(con.contour)
                    #to_draw_approx.append(con.approx)
                else:
                    to_draw.append(con.contour)
                    to_draw_approx.append(con.approx)

            tmp_image = image.copy()
            # cons that are squares are in green
            # for non-squqres the approx is green and contour is blue
            cv2.drawContours(tmp_image, to_draw, -1, (255, 0, 0), 2)
            cv2.drawContours(tmp_image, to_draw_approx, -1, (0, 0, 255), 2)
            cv2.drawContours(tmp_image, to_draw_square, -1, (0, 255, 0), 2)

            if to_draw_missing:
                cv2.drawContours(tmp_image, to_draw_missing, -1, (0, 255, 255), 2)
                cv2.drawContours(tmp_image, to_draw_missing_approx, -1, (255, 255, 0), 2)

            cv2.imshow(desc, tmp_image)
            cv2.waitKey(0)
        else:
            cv2.imshow(desc, image)
            cv2.waitKey(0)

    def get_contour_neighbors(self, contours, target_con, strict=True):
        """
        Return stats on how many other contours are in
        the same 'row' or 'col' as target_con

        ROW_THRESHOLD determines how far up/down we look for other contours
        COL_THRESHOLD determines how far left/right we look for other contours
        """
        row_neighbors = 0
        row_square_neighbors = 0
        col_neighbors = 0
        col_square_neighbors = 0

        # These are percentages of the image width and height
        if strict:
            ROW_THRESHOLD = 0.04
            COL_THRESHOLD = 0.04
        else:
            # dwalton this needs some work....it really depends on the
            # cube size...the smaller the cube the larger the threshold
            ROW_THRESHOLD = 0.06
            COL_THRESHOLD = 0.06

        width_wiggle = int(self.img_width * COL_THRESHOLD)
        height_wiggle = int(self.img_height * ROW_THRESHOLD)

        target_cX = target_con.cX
        target_cY = target_con.cY

        log.debug("get_contour_neighbors() for %s, width_wiggle %s, height_wiggle %s" %
            (target_con, width_wiggle, height_wiggle))

        for con in contours:

            # do not count yourself
            if con == target_con:
                continue

            x_delta = abs(con.cX - target_cX)
            y_delta = abs(con.cY - target_cY)

            if x_delta <= width_wiggle:
                col_neighbors += 1

                if con.is_square(self.square_area):
                    col_square_neighbors += 1
                    log.debug("%s is a square col neighbor" % con)
                else:
                    log.debug("%s is a col neighbor but has %d corners" % (con, con.corners))
            else:
                log.debug("%s delta %s is outside width wiggle room %s" % (con, x_delta, width_wiggle))

            if y_delta <= height_wiggle:
                row_neighbors += 1

                if con.is_square(self.square_area):
                    row_square_neighbors += 1
                    log.debug("%s is a square row neighbor" % con)
                else:
                    log.debug("%s is a row neighbor but has %d corners" % (con, con.corners))
            else:
                log.debug("%s delta %s is outside height wiggle room %s" % (con, y_delta, height_wiggle))

        log.debug("get_contour_neighbors() for contour (%d, %d) has row %d, row_square %d, col %d, col_square %d neighbors\n" %
            (target_cX, target_cY, row_neighbors, row_square_neighbors, col_neighbors, col_square_neighbors))

        return (row_neighbors, row_square_neighbors, col_neighbors, col_square_neighbors)

    def sort_by_row_col(self, contours, size):
        """
        Given a list of contours sort them starting from the upper left corner
        and ending at the bottom right corner
        """
        result = []
        num_squares = len(contours)

        for row_index in xrange(size):

            # We want the 'size' squares that are closest to the top
            tmp = []
            for con in contours:
                tmp.append((con.cY, con.cX))
            top_row = sorted(tmp)[:size]

            # Now that we have those, sort them from left to right
            top_row_left_right = []
            for (cY, cX) in top_row:
                top_row_left_right.append((cX, cY))
            top_row_left_right = sorted(top_row_left_right)

            log.info("sort_by_row_col() row %d: %s" % (row_index, pformat(top_row_left_right)))
            contours_to_remove = []
            for (target_cX, target_cY) in top_row_left_right:
                for con in contours:

                    if con in contours_to_remove:
                        continue

                    if con.cX == target_cX and con.cY == target_cY:
                        result.append(con)
                        contours_to_remove.append(con)
                        break

            for con in contours_to_remove:
                contours.remove(con)

        assert len(result) == num_squares, "sort_by_row_col is returning %d squares, it should be %d" % (len(result), num_squares)
        return result

    def remove_non_square_contours(self):
        """
        Remove non-square contours from candidates.  Return a list of the ones we removed.
        """
        candidates_to_remove = []

        # Remove parents with square child contours
        for con in self.candidates:
            if not con.is_square():
                candidates_to_remove.append(con)

        for x in candidates_to_remove:
            self.candidates.remove(x)

        return candidates_to_remove

    def remove_square_within_square_contours(self):
        candidates_to_remove = []

        # Remove parents with square child contours
        for con in self.candidates:
            if con.is_square() and con.child_is_square():
                candidates_to_remove.append(con)

        # Remove contours whose parents are square
        for con in self.candidates:
            if con in candidates_to_remove:
                continue

            if con.parent_is_square():
                parent = con.get_parent()

                if parent not in candidates_to_remove:
                    candidates_to_remove.append(con)

        for x in candidates_to_remove:
            self.candidates.remove(x)

        removed = len(candidates_to_remove)
        log.info("remove_square_within_square_contours() %d removed, %d remain" % (removed, len(self.candidates)))
        return True if removed else False

    def get_cube_size(self):
        """
        Look at all of the contours that are squares and see how many square
        neighbors they have in their row and col. Store the number of square
        contours in each row/col in data, then sort data and return the
        median entry
        """
        candidates = deepcopy(self.candidates)

        # Find the median area of all square contours and find the top, right,
        # bottom, left boundry of all square contours
        square_areas = []
        for con in candidates:
            if con.is_square():
                square_areas.append(int(con.area))

        square_areas = sorted(square_areas)
        num_squares = len(square_areas)
        square_area_index = int(num_squares/2)
        self.square_area = int(square_areas[square_area_index])

        log.info("%d squares, median index %d, median area %d, all square areas %s" %\
            (num_squares, square_area_index, self.square_area, ','.join(map(str, square_areas))))

        # Now find all of the square contours that are the same size (roughly) as the
        # median square size. Look to see how many square neighbors are in the row and
        # col for each square contour.
        data = []
        top = None
        right = None
        bottom = None
        left = None

        for con in self.candidates:
            if con.is_square(self.square_area):
                (row_neighbors, row_square_neighbors, col_neighbors, col_square_neighbors) =\
                    self.get_contour_neighbors(self.candidates, con)
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

        self.size = median_size
        self.top = top
        self.right = right
        self.bottom = bottom
        self.left = left

    def remove_contours_outside_cube(self, contours):
        contours_to_remove = []

        # These are percentages of the image width and height
        ROW_THRESHOLD = 0.12
        COL_THRESHOLD = 0.10

        top    = self.top    * (1 - ROW_THRESHOLD)
        bottom = self.bottom * (1 + ROW_THRESHOLD)
        left   = self.left   * (1 - COL_THRESHOLD)
        right  = self.right  * (1 + COL_THRESHOLD)

        for con in contours:
            if con.cY < top or con.cY > bottom or con.cX < left or con.cX > right:
                contours_to_remove.append(con)

        for con in contours_to_remove:
            contours.remove(con)

        removed = len(contours_to_remove)
        log.info("remove_contours_outside_cube() %d removed, %d remain" % (removed, len(contours)))
        return True if removed else False

    def remove_small_squares(self):
        """
        Remove squares that are much smaller than self.square_area
        """
        candidates_to_remove = []

        for con in self.candidates:
            if not con.is_square(self.square_area):
                candidates_to_remove.append(con)
                log.info("remove_small_squares() %s area %d not close to median area %d" % (con, con.area, self.square_area))

        for con in candidates_to_remove:
            self.candidates.remove(con)

        removed = len(candidates_to_remove)
        log.info("remove_small_squares() %d removed, %d remain" % (removed, len(self.candidates)))
        return True if removed else False

    def sanity_check_results(self, contours, strict, use_assert, debug=False):
        """
        """

        # Verify we found the correct number of squares
        num_squares = len(contours)
        needed_squares = self.size * self.size

        if num_squares != needed_squares:
            if use_assert:
                assert False, "Should have found %s squares, we found %d" % (needed_squares, num_squares)
            else:
                if debug:
                    log.info("sanity_check_results() False: num_squares %d != needed_squares %d" % (num_squares, needed_squares))
                return False

        # Verify each row/col has the same number of neighbors
        req_neighbors = self.size - 1

        for con in contours:
            (row_neighbors, row_square_neighbors, col_neighbors, col_square_neighbors) =\
                self.get_contour_neighbors(contours, con, strict)

            if row_neighbors != req_neighbors:
                if use_assert:
                    assert False, "%s has %d row neighbors, must be %d" % (con, row_neighbors, req_neighbors)
                else:
                    if debug:
                        log.info("sanity_check_results() False: row_neighbors %d != req_neighbors %s" % (row_neighbors, req_neighbors))
                    return False

            if col_neighbors != req_neighbors:
                if use_assert:
                    assert False, "%s has %d col neighbors, must be %d" % (con, col_neighbors, req_neighbors)
                else:
                    if debug:
                        log.info("sanity_check_results() False: col_neighbors %d != req_neighbors %s" % (col_neighbors, req_neighbors))
                    return False

        return True

    def find_missing_squares(self):

        # How many squares are missing?
        missing_count = (self.size * self.size) - len(self.candidates)
        missing = []

        if missing_count:

            # Of the non-square contours that we previously removed, ignore the ones that are outside the cube
            self.remove_contours_outside_cube(self.non_square_contours)
            log.info("find_missing_squares() %d squares are missing, there are %d non-square contours inside the cube" %
                     (missing_count, len(self.non_square_contours)))

            missing_candidates = []

            # Try all permutations of the non-square contours inside the cube until
            # we find the combination of non_square_contours to add that results in
            # a valid cube. There could be multiple permutations that satisfy this
            # requirement, use the one whose contours result in the largest area.
            for combo in combinations(self.non_square_contours, missing_count):

                #if combo[0].index != 58:
                #    continue
                #log.warning("HERE 10 %s" % combo[0])

                tmp_candidates = deepcopy(self.candidates)
                tmp_candidates.extend(combo)

                if self.sanity_check_results(tmp_candidates, strict=False, use_assert=False):
                    combo_area = 0
                    for tmp in combo:
                        combo_area += tmp.area
                    missing_candidates.append((combo_area, combo))

            missing_candidates = list(reversed(sorted(missing_candidates)))

            if missing_candidates:
                missing = missing_candidates[0][1]
            else:
                raise Exception("Could not find missing squares needed to create a valid cube")

        return missing

    def analyze(self):
        log.warning("Analyze %s" % self.filename)
        self.image = cv2.imread(self.filename)
        (self.img_height, self.img_width) = self.image.shape[:2]

        # convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # self.draw_cube(gray, None, "00 gray")

        # blur a little...not sure why but most canny examples I've found do this prior to running canny
        # http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        # self.draw_cube(blurred, None, "10 blurred")

        # canny to find the edges
        canny = cv2.Canny(blurred, 20, 40)
        # self.draw_cube(canny, None, "20 canny")

        # dilate the image to make the edge lines thicker
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(canny, kernel, iterations=2)
        self.draw_cube(dilated, None, "30 dilated")

        # find the contours and create a CustomContour object for each...store these in "candidates"
        # http://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html
        # http://stackoverflow.com/questions/11782147/python-opencv-contour-tree-hierarchy
        (contours, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.candidates = []
        hierarchy = hierarchy[0]

        index = 0
        for component in zip(contours, hierarchy):
            con = CustomContour(self, index, component[0], component[1])
            self.contours_by_index[index] = con

            if con.area > 30 and con.cX is not None:
                self.candidates.append(con)
            index += 1

        # Throw away the contours that do not look like squares
        self.draw_cube(self.image, "40 pre non-squares removal #1")
        self.non_square_contours = self.remove_non_square_contours()
        self.draw_cube(self.image, "50 post non-squares removal #1")

        # Sometimes we find a square within a square due to the black space
        # between the cube squares.  Throw away the outside square (it contains
        # the black edge) and keep the inside square.
        if self.remove_square_within_square_contours():
            self.draw_cube(self.image, "60 post square-within-square removal #1")

        # Find the cube size and extreme coordinates of all known squares
        self.get_cube_size()

        # remove all contours outside those coordinates
        self.remove_contours_outside_cube(self.candidates)
        self.draw_cube(self.image, "70 post outside cube removal")

        missing = []

        if not self.sanity_check_results(self.candidates, strict=True, use_assert=False):
            # remove any squares within the cube that are so small they are obviously not cube squares
            if self.remove_small_squares():
                self.draw_cube(self.image, "80 post small square removal")

            if not self.sanity_check_results(self.candidates, strict=True, use_assert=False):
                missing = self.find_missing_squares()
                self.candidates.extend(missing)

        self.draw_cube(self.image, "90 Final", missing)

        raw_data = []
        for con in self.sort_by_row_col(deepcopy(self.candidates), self.size):
            # TODO - this needs to be the median (midpoint) since sometime we get the black
            # edge between squares...that will make the mean darker than we want

            # Use the mean value of the contour
            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [con.contour], 0, 255, -1)
            (mean_blue, mean_green, mean_red, _)= map(int, cv2.mean(self.image, mask = mask))
            raw_data.append((mean_red, mean_green, mean_blue))
        log.info("squares RGB data\n%s\n" % pformat(raw_data))

        squares_per_side = len(raw_data)
        size = int(math.sqrt(squares_per_side))
        init_square_index = (self.index * squares_per_side) + 1

        square_indexes = []
        for row in range(size):
            square_indexes_for_row = []
            for col in range(size):
                square_indexes_for_row.append(init_square_index + (row * size) + col)
            square_indexes.append(square_indexes_for_row)

        '''
        TODO clean this up

        The L, F, R, and B sides are simple, for the U and D sides the cube in
        the png is rotated by 90 degrees so we need to rotate our array of
        square indexes by 90 degrees to compensate
        '''
        if self.name in ('U', 'D'):
            my_indexes = rotate_2d_array(square_indexes)
        else:
            my_indexes = square_indexes

        log.info("%s square_indexes\n%s\n" % (self, pformat(square_indexes)))
        log.info("%s my_indexes\n%s\n" % (self, pformat(my_indexes)))
        my_indexes = compress_2d_array(my_indexes)
        log.info("%s my_indexes (final) %s" % (self, str(my_indexes)))

        for index in range(squares_per_side):
            square_index = my_indexes[index]
            (red, green, blue) = raw_data[index]
            log.info("square %d RGB (%d, %d, %d)" % (square_index, red, green, blue))

            # self.data is a dict where the square number (as an int) will be
            # the key and a RGB tuple the value
            self.data[square_index] = (red, green, blue)

        log.info("\n\n\n")
