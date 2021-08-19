import math
import os
import random as rng
import sys
from typing import Iterator

import cv2 as cv
import gpxpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lp_cascade import Cascader

DEG2RAD = math.pi / 180


class Lamppost:
    """
    Class for tracking lampposts.
    """

    def __init__(self, x: int, y: int, w: int, h: int, lp_id: int, color: tuple, frame: np.ndarray, gps: tuple):
        """
        Constructor method.

        :param x: x coordinate of bounding box
        :param y: y coordinate of bounding box
        :param w: width of bounding box
        :param h: height of bounding box
        :param lp_id: id of lamppost
        :param color: color of its bounding box in (r,g,b) format
        :param frame: frame in which the lamppost was found
        """

        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.lp_id = lp_id
        self.color = color

        # how many frames this lamppost has not been detected
        self.decay = 0
        # real world coordinates where this lamppost will be detected
        self.locs = [gps]
        # angles between camera and lamppost during detection
        self.rads = []
        self.get_angle()
        self.uv = None
        self.intersect()

        self.tracker = cv.TrackerKCF_create()
        self.tracker.init(frame, (x, y, w, h))

    def update_bbox(self, x: int, y: int, w: int, h: int):
        """
        Update bounding box of lamppost.

        :param x: x coordinate of bounding box
        :param y: y coordinate of bounding box
        :param w: width of bounding box
        :param h: height of bounding box
        """

        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def get_bbox(self) -> tuple:
        """
        Returns bounding box of lamppost on the image plane in a
        (x, y, w, h) tuple.

        :return: tuple of coordinates
        :rtype: tuple
        """

        return (self.x, self.y), (self.x + self.w, self.y + self.h)

    def get_coor(self) -> tuple:
        """
        Returns the x, y coordinates of the lamppost on the image plane.

        :return: x, y coordinates of lamppost.
        :rtype: tuple
        """

        return self.x, self.y - 10

    def get_dist_loc(self) -> tuple:
        """
        Return location for when printing distance

        :return: x, y tuple
        :rtype: tuple
        """

        return self.x, self.y + self.h + 30

    def get_id(self) -> int:
        """
        Returns id of lamppost.

        :return: id of lamppost.
        :rtype: int
        """

        return self.lp_id

    def get_color(self) -> tuple:
        """
        Returns the color associated with its bounding box.

        :return: tuple of (r,g,b) color.
        :rtype: tuple
        """

        return self.color

    def get_decay(self) -> int:
        """
        Returns decay of lamppost.

        :return: decay of lamppost.
        :rtype: int
        """

        return self.decay

    def reset_decay(self):
        """
        Resets the decay of lamppost.
        """
        self.decay = 0

    def inc_decay(self):
        """
        Increments decay of lamppost.
        """

        self.decay += 1

    def tracker_update(self, frame: np.ndarray):
        """
        Update tracker with new frame.

        :param frame: frame which to update
        :return: boolean indicating success
        """

        success, (x, y, w, h) = self.tracker.update(frame)
        if success:
            self.update_bbox(x, y, w, h)
        # print(success)
        return success

    def get_angle(self) -> float:
        """
        Returns the angle between camera and lamppost in radians according to
        the x coordinate of the bounding box.

        formula: result = ((x - minX) / (maxX - minX)) * (maxFOV - minFOV) + minFOV

        :return: angle between camera in lamppost in radians
        :rtype: float
        """

        deg = ((self.x + (self.w * 0.5)) / 1920) * 60 - 30
        rad = deg * DEG2RAD
        self.rads.append(rad)
        return rad

    def add_loc(self, gps: tuple):
        """
        Adds gps coordinates at which this lamppost was detected
        to a list. gps tuple must be in the form of (latitude, longitude).

        :param gps: gps coordinate (latitude, longitude)
        """
        self.locs.append(gps)

    def intersect(self) -> np.ndarray:
        """
        Calculate the least squares solution of the point closest
        to all lines defined by the gps positions and the angles
        to the lamppost. https://stackoverflow.com/a/52089867

        :return: point of (nearest) intersection
        :rtype: np.ndarray
        """
        dir_vectors = np.asarray([[math.cos(rad), math.sin(rad)] for rad in self.rads])
        self.uv = dir_vectors / np.sqrt((dir_vectors ** 2).sum(-1))[..., np.newaxis]

        if len(self.locs) < 2:
            return (np.array([]), np.array([]))

        points = np.asarray(self.locs)

        projs = np.eye(self.uv.shape[1]) - self.uv[:, :, np.newaxis] * self.uv[:, np.newaxis]  # I - n*n.T

        R = projs.sum(axis=0)
        q = (projs @ points[:, :, np.newaxis]).sum(axis=0)

        p = np.linalg.lstsq(R, q, rcond=-1)[0]

        return p

    def calc_dist(self) -> float:
        """
        Given two or more angles and gps coordinates, calculate
        the vectors belonging to each angle and their intersection

        :return: distance to lamppost
        """

        if len(self.locs) < 2:
            return np.inf

        dir_vectors = np.asarray([[math.cos(rad), math.sin(rad)] for rad in self.rads])
        uni_vectors = dir_vectors / np.sqrt((dir_vectors ** 2).sum(-1))[..., np.newaxis]
        org_vectors = np.asarray(self.locs)

        proj = np.eye(2) - uni_vectors[:, np.newaxis]


class Lp_container:
    """
    Container for the Lamppost objects.
    """

    def __init__(self):
        """
        Constructor method.
        """
        self.lps = []
        self.det_lps = set()

        # the amount of frames a lamppost is allowed to decay
        self.max_decay = 12

    def add_lp(self, lp: Lamppost) -> None:
        """
        Adds lampposts to container

        :param lp: lamppost to add
        """
        self.lps.append(lp)

    def find_matching_lp(self, det_lp: tuple) -> Lamppost:
        """
        Finds the closest matching known lamppost to the newly
        detected lamppost according to their location on the
        image plane or None if there is no match.

        :param det_lp: detected lamppost to check against know lampposts
        :return: closest matching lamppost object or None if none are found
        :rtype: Lamppost
        """

        # the max distance in pixels for it to be a valid match, doesn't work
        max_dist = 1000

        if len(self.lps) == 0:
            return None

        # generate a list with the coordinates of all know lampposts on the image plane
        coors = np.asarray([lp.get_coor() for lp in self.lps])

        dists = np.sum((coors - np.asarray(det_lp)) ** 2, axis=1)

        if np.amin(dists) < max_dist:
            lp = self.lps[np.argmin(dists)]
            self.det_lps.add(lp)
            return lp

    def apply_decay(self) -> None:
        """
        Resets decay on detected lampposts and increments them
        on those not detected. Removes lamppost object from
        container if it reaches max_decay.
        """

        # detected lampposts have their decay reset
        for lp in self.det_lps:
            lp.reset_decay()

        decaying_lps = set(self.lps).difference(self.det_lps)

        # apply decay
        for lp in decaying_lps:
            lp.inc_decay()
            # remove lamppost who have exceeded max_decay
            if lp.get_decay() > self.max_decay:
                self.lps.remove(lp)

        self.det_lps.clear()

    def get_lps(self) -> list:
        """
        Returns all currently tracked lampposts

        :return: list with lampposts
        :rtype: list
        """
        return self.lps

    def del_lp(self, lp: Lamppost):
        """
        Remove lamppost from currently tracked lampposts

        :param lp: lamppost to remove
        """

        self.lps.remove(lp)


def number_gen() -> Iterator[int]:
    """
    Natural number generator

    :return: the next integer in the sequence of natural numbers
    :rtype: Iterator[int]
    """
    num = 0
    while True:
        yield num
        num += 1


def color_gen() -> tuple:
    """
    Random (r,g,b) color generator

    :return: tuple of (r,g,b)
    :rtype: tuple
    """
    return tuple(np.random.choice(range(256), size=3).tolist())


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6372.8  # Radius of earth in kilometers.
    return c * r * 1000

cv2 = cv

X0 = 155000
Y0 = 463000
PHI0 = 52.15517440
LAM0 = 5.38720621


def rd_to_wgs(x, y):
    """
    Convert rijksdriehoekcoordinates into WGS84 coordinates. Input parameters: x (float), y (float).
    """

    if isinstance(x, (list, tuple)):
        x, y = x

    pqk = [(0, 1, 3235.65389),
           (2, 0, -32.58297),
           (0, 2, -0.24750),
           (2, 1, -0.84978),
           (0, 3, -0.06550),
           (2, 2, -0.01709),
           (1, 0, -0.00738),
           (4, 0, 0.00530),
           (2, 3, -0.00039),
           (4, 1, 0.00033),
           (1, 1, -0.00012)]

    pql = [(1, 0, 5260.52916),
           (1, 1, 105.94684),
           (1, 2, 2.45656),
           (3, 0, -0.81885),
           (1, 3, 0.05594),
           (3, 1, -0.05607),
           (0, 1, 0.01199),
           (3, 2, -0.00256),
           (1, 4, 0.00128),
           (0, 2, 0.00022),
           (2, 0, -0.00022),
           (5, 0, 0.00026)]

    dx = 1E-5 * (x - X0)
    dy = 1E-5 * (y - Y0)

    phi = PHI0
    lam = LAM0

    for p, q, k in pqk:
        phi += k * dx ** p * dy ** q / 3600

    for p, q, l in pql:
        lam += l * dx ** p * dy ** q / 3600

    return [phi, lam]


def wgs_to_rd(phi, lam):
    """
    Convert WGS84 coordinates into rijksdriehoekcoordinates. Input parameters: phi (float), lambda (float).
    """

    pqr = [(0, 1, 190094.945),
           (1, 1, -11832.228),
           (2, 1, -114.221),
           (0, 3, -32.391),
           (1, 0, -0.705),
           (3, 1, -2.34),
           (1, 3, -0.608),
           (0, 2, -0.008),
           (2, 3, 0.148)]

    pqs = [(1, 0, 309056.544),
           (0, 2, 3638.893),
           (2, 0, 73.077),
           (1, 2, -157.984),
           (3, 0, 59.788),
           (0, 1, 0.433),
           (2, 2, -6.439),
           (1, 1, -0.032),
           (0, 4, 0.092),
           (1, 4, -0.054)]

    dphi = 0.36 * (phi - PHI0)
    dlam = 0.36 * (lam - LAM0)

    X = X0
    Y = Y0

    for p, q, r in pqr:
        X += r * dphi ** p * dlam ** q

    for p, q, s in pqs:
        Y += s * dphi ** p * dlam ** q

    return [X, Y]


def detect_frame(im):
    img = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    img = cv.medianBlur(img, 5)
    kernel = np.ones((12, 12), np.uint8)
    img = cv.erode(img, kernel, iterations=3)
    img = cv.dilate(img, kernel, iterations=3)

    # cv.imshow("AAAAAAAA", img)

    perc = 0.01
    hist = cv.calcHist([img[:324]], [0], None, [256], [0, 256]).flatten()

    total = img.shape[0] * img.shape[1]
    target = perc * total
    # print(target)
    summed = 0
    thresh = 0
    for i in range(255, 0, -1):
        summed += int(hist[i])
        if summed >= target:
            thresh = i
            break

    ret = cv.threshold(img, thresh < 255 and thresh or 254, 0, cv.THRESH_TOZERO)[1]

    contours = cv2.findContours(ret[:, :960], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])

    for i in range(len(contours)):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        color = (0, 255, 0)
        if centers[i][1] <= 0.3 * 1080 and radius[i] < 100:
            cv.rectangle(im, (int(boundRect[i][0]), int(boundRect[i][1])),
                         (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)

    return boundRect


def demo():
    lp_container = Lp_container()
    cas = Cascader("models/cascade.xml")
    id_gen = number_gen()
    cap = cv.VideoCapture("input/Amsterdam/AMSTERDAM_OSV.mp4")
    idx = 0

    with open("input/Amsterdam/AMSTERDAM_OSV.gpx") as f:
        gpx = gpxpy.parse(f)

    segments = gpx.tracks[0].segments[0]
    gps_coords = pd.DataFrame([
        {'lat': p.latitude,
         'lon': p.longitude,
         'ele': p.elevation,
         'time': p.time} for p in segments.points])

    lats = [gps_coords.lat[0]]
    lons = [gps_coords.lon[0]]

    figure, ax = plt.subplots(figsize=(8, 6))

    while True:
        _, frame = cap.read()
        rdx, rdy = wgs_to_rd(lats[idx % 24], lons[idx % 24])
        if idx % 24 == 0:
            lp_coors = cas.cascade_frame(frame)
            # lp_coors = detect_frame(frame)
            lats = np.linspace(gps_coords.lat[math.floor(idx / 24)], gps_coords.lat[math.floor(idx / 24 + 1)], num=24)
            lons = np.linspace(gps_coords.lon[math.floor(idx / 24)], gps_coords.lon[math.floor(idx / 24 + 1)], num=24)

            for x, y, w, h in lp_coors:
                match = lp_container.find_matching_lp((x, y))

                if not match:
                    lp = Lamppost(x, y, w, h, next(id_gen), color_gen(), frame, (rdx, rdy))
                    lp_container.add_lp(lp)
                else:
                    match.update_bbox(x, y, w, h)
                    match.add_loc((rdx, rdy))
                    match.get_angle()
        else:
            for lp in lp_container.get_lps():
                if not lp.tracker_update(frame):
                    # lp_container.del_lp(lp)
                    lp.inc_decay()
                else:
                    lp_container.find_matching_lp(lp.get_coor())
                lp.add_loc((rdx, rdy))
                lp.get_angle()

        for lp in lp_container.get_lps():
            p0, p1 = lp.get_bbox()
            frame = cv.rectangle(frame, p0, p1, lp.get_color(), 4)
            frame = cv.putText(frame,
                               text=f"id:{lp.get_id()}",
                               org=lp.get_coor(),
                               fontFace=0,
                               fontScale=1,
                               color=lp.get_color(),
                               thickness=3)
            lp_rdx, lp_rdy = lp.intersect()
            print(lp_rdx, lp_rdy)

            if lp_rdx.size == 0:
                continue

            # dist = haversine(lons[idx % 24], lats[idx % 24], lon2, lat2)
            dist = round(np.sqrt((lp_rdx - rdx)**2 + (lp_rdy - rdy)**2)[0] / 10, 2)
            frame = cv.putText(frame,
                               text=f"dist: {dist}",
                               org=lp.get_dist_loc(),
                               fontFace=0,
                               fontScale=1,
                               color=lp.get_color(),
                               thickness=3)

        frame = cv.putText(frame,
                           text=f"lat: {lats[idx % 24]}",
                           org=(1500, 1040),
                           fontFace=0,
                           fontScale=1,
                           color=(255, 255, 255),
                           thickness=2)
        frame = cv.putText(frame,
                           text=f"lon: {lons[idx % 24]}",
                           org=(1500, 1070),
                           fontFace=0,
                           fontScale=1,
                           color=(255, 255, 255),
                           thickness=2)

        # frame = cv.line(frame, (960, 0), (960, 1080), (255, 255, 255), 5)
        cv.imshow("tracking", frame)
        # print(lp_container.get_lps()[0].uv)
        # quiv = ax.quiver(lp_container.get_lps()[0].locs,lp_container.get_lps()[0].uv )
        # figure.canvas.draw()
        k = cv.waitKey(0)
        # figure.canvas.flush_events()

        if k == ord("q"):
            cv.destroyWindow("tracking")
            sys.exit(0)

        lp_container.apply_decay()
        # print([(lp.get_id(), lp.get_decay()) for lp in lp_container.get_lps()])
        idx += 1


if __name__ == "__main__":
    demo()
