import math
import os
import random as rng
import sys


import cv2 as cv
import gpxpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lp_cascade import Cascader
from utils import number_gen, color_gen, wgs_to_rd, detect_frame

DEG2RAD = math.pi / 180
RAD2DEG = 180 / math.pi
X0 = 155000
Y0 = 463000
PHI0 = 52.15517440
LAM0 = 5.38720621


class Lamppost:
    """
    Class for tracking lampposts.
    """

    def __init__(self, x: int, y: int, w: int, h: int, lp_id: int, color: tuple, frame: np.ndarray, gps: tuple, bearing):
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
        self.get_angle(bearing)
        self.uv = None

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
        return success

    def get_angle(self) -> tuple:
        """
        Returns the angle between camera and lamppost in radians according to
        the x and y coordinates  of the bounding box.

        formula: result = ((x - minX) / (maxX - minX)) * (maxFOV - minFOV) + minFOV

        :return: angle between camera in lamppost in radians in (x, y) format
        :rtype: tuple
        """
        deg_alpha = bearing - ((self.x + (self.w * 0.5)) / 1920) * 62.8 - 31.9
        deg_beta = (((1080 - self.y) + (self.h * 0.5)) / 1080) * 36.2 - 18.1
        alpha = deg_alpha * DEG2RAD
        beta = deg_beta * DEG2RAD
        self.rads.append((alpha, beta))
        return alpha, beta

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
        print(self.get_id())

        if len(self.locs) < 2:
            return np.array([[np.nan], [np.nan], [np.nan]])

        points = np.asarray(self.locs)


        dir_vector = np.asarray([[math.cos(alpha) * math.cos(beta),
                                  math.sin(alpha) * math.cos(beta),
                                  math.sin(beta)] for alpha, beta in self.rads])

        self.uv = dir_vector / np.sqrt((dir_vector ** 2).sum(-1))[..., np.newaxis]

        projs = np.eye(self.uv.shape[1]) - self.uv[:, :, np.newaxis] * self.uv[:, np.newaxis]

        R = projs.sum(axis=0)
        q = (projs @ points[:, :, np.newaxis]).sum(axis=0)

        ls = np.linalg.lstsq(R, q, rcond=-1)
        p = ls[0]
        self.point_line_distance(p)
        return p

    def point_line_distance(self, point):
        point = np.reshape(point, (1, 3))
        P0 = np.asarray(self.locs)
        P1 = P0 + self.uv
        dist = 0

        for i in range(len(P0)):
            d = np.reshape((P1[i] - P0[i]) / np.linalg.norm(P1[i] - P0[i]), (3))
            v = np.reshape(point - P0[i], (3))
            t = np.dot(v, d.T)
            P = P0[i] + np.dot(t, d.T)
            dist += np.linalg.norm(P - point)
        print(dist / len(P0))
        print()





    # def intersect(self):
    #     """P0 and P1 are NxD arrays defining N lines.
    #     D is the dimension of the space. This function
    #     returns the least squares intersection of the N
    #     lines from the system given by eq. 13 in
    #     http://cal.cs.illinois.edu/~johannes/research/LS_line_intersect.pdf.
    #     """
    #     # generate all line direction vectors
    #
    #     if len(self.locs) < 2:
    #         return np.array([[np.nan], [np.nan], [np.nan]])
    #
    #     dir_vector = np.asarray([[math.cos(alpha) * math.cos(beta),
    #                               math.sin(alpha) * math.cos(beta),
    #                               math.sin(beta)] for alpha, beta in self.rads])
    #
    #     self.uv = dir_vector / np.sqrt((dir_vector ** 2).sum(-1))[..., np.newaxis]
    #
    #     P0 = np.asarray(self.locs)
    #     P1 = P0 + self.uv
    #     n = (P1-P0)/np.linalg.norm(P1-P0,axis=1)[:,np.newaxis] # normalized
    #
    #     # generate the array of all projectors
    #     projs = np.eye(n.shape[1]) - n[:,:,np.newaxis]*n[:,np.newaxis]  # I - n*n.T
    #     # see fig. 1
    #
    #     # generate R matrix and q vector
    #     R = projs.sum(axis=0)
    #     q = (projs @ P0[:,:,np.newaxis]).sum(axis=0)
    #
    #     # solve the least squares problem for the
    #     # intersection point p: Rp = q
    #     p = np.linalg.lstsq(R,q,rcond=None)[0]
    #     print(p)
    #
    #     return p

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


def demo():
    lp_container = Lp_container()
    cas = Cascader("models/cascade.xml")
    id_gen = number_gen()
    cap = cv.VideoCapture("input/Amsterdam/AMSTERDAM_OSV.mp4")

    with open("input/Amsterdam/AMSTERDAM_OSV.gpx") as f:
        gpx = gpxpy.parse(f)

    segments = gpx.tracks[0].segments[0]
    gps_coords = pd.DataFrame([
        {'lat': p.latitude,
         'lon': p.longitude,
         'ele': p.elevation,
         'time': p.time} for p in segments.points])

    lats = [gps_coords.lat[0], gps_coords.lat[1]]
    lons = [gps_coords.lon[0], gps_coords.lon[1]]

    # estimated height of the camera on the bike in meter
    rdy = 1.2

    idx = 0
    while True:
        _, frame = cap.read()
        rdx, rdz = wgs_to_rd(lats[idx % 24], lons[idx % 24])

        if idx % 24 == 0:
            lp_coors = cas.cascade_frame(frame)
            # lp_coors = detect_frame(frame)

            # interpolate the coordinates between two data-points
            lats = np.linspace(gps_coords.lat[math.floor(idx / 24)], gps_coords.lat[math.floor(idx / 24 + 1)], num=24)
            lons = np.linspace(gps_coords.lon[math.floor(idx / 24)], gps_coords.lon[math.floor(idx / 24 + 1)], num=24)

            p0 = wgs_to_rd(lats[0],  lons[0])
            p1 = wgs_to_rd(lats[-1], lons[-1])
            d = round(np.sqrt((p1[1] - p0[1])**2 + (p1[0] - p0[0])**2))
            # print("m/s:", d)
            bearing = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
            bearing = bearing * RAD2DEG

            for x, y, w, h in lp_coors:
                match = lp_container.find_matching_lp((x, y))

                if not match:
                    lp = Lamppost(x, y, w, h, next(id_gen), color_gen(), frame, (rdx, rdz, rdy), bearing)
                    lp_container.add_lp(lp)

                else:
                    match.update_bbox(x, y, w, h)
                    match.add_loc((rdx, rdz, rdy))
                    match.get_angle(bearing)

        else:
            for lp in lp_container.get_lps():
                if not lp.tracker_update(frame):
                    # lp_container.del_lp(lp)
                    lp.inc_decay()

                else:
                    lp_container.find_matching_lp(lp.get_coor())

                lp.add_loc((rdx, rdz, rdy))
                lp.get_angle(bearing)

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

            lp_rdx, lp_rdz, lp_rdy = lp.intersect()

            if np.isnan(lp_rdx):
                continue

            dist = round(np.sqrt((lp_rdx - rdx) ** 2 + (lp_rdz - rdz) ** 2 + (lp_rdy - rdy) ** 2)[0] / 10, 2)
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

        cv.imshow("tracking", frame)
        k = cv.waitKey(0)

        if k == ord("q"):
            cv.destroyWindow("tracking")
            sys.exit(0)

        lp_container.apply_decay()
        idx += 1


if __name__ == "__main__":
    demo()
