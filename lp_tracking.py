import os
from typing import Iterator
import cv2 as cv
import numpy as np
from lp_cascade import Cascader
import sys
import gpxpy
import pandas as pd
import math


class Lamppost:
    """
    Class for tracking lampposts.
    """

    def __init__(self, x: int, y: int, w: int, h: int, lp_id: int, color: tuple, frame: np.ndarray):
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
        # self.tracker = cv.TrackerMIL_create()
        # self.tracker.init()

        self.tracker = cv.TrackerKCF_create()
        self.tracker.init(frame, (x, y, w, h))

    def update_location(self, x: int, y: int, w: int, h: int) -> None:
        """
        Update location of lamppost.

        :param x: x coordinate of bounding box
        :param y: y coordinate of bounding box
        :param w: width of bounding box
        :param h: height of bounding box
        :return: None
        """

        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def get_box(self) -> tuple:
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

        :return: x, y coordinates of lamppost
        :rtype: tuple
        """
        return self.x, self.y

    def get_id(self) -> int:
        """
        Returns id of lamppost

        :return: id of lamppost
        :rtype: int
        """
        return self.lp_id

    def get_color(self) -> tuple:
        """
        Returns the color associated with its bounding box

        :return: tuple of (r,g,b) color
        :rtype: tuple
        """
        return self.color

    def get_decay(self) -> int:
        """
        Returns decay of lamppost

        :return: decay of lamppost
        :rtype: int
        """
        return self.decay

    def reset_decay(self):
        """
        Resets the decay of lamppost

        :return:
        """
        self.decay = 0

    def inc_decay(self):
        """
        Increments decay of lamppost

        :return:
        """
        self.decay += 1

    def tracker_update(self, frame: np.ndarray):
        """
        Update tracker with new frame

        :param frame: frame which to update
        :return: boolean indicating success
        """

        success, (x, y, w, h) = self.tracker.update(frame)
        if success:
            self.update_location(x, y, w, h)
        # print(success)
        return success


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
        self.max_decay = 24

    def add_lp(self, lp: Lamppost) -> None:
        """
        Adds lampposts to container

        :param lp: lamppost to add
        :return: None
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
        max_dist = 100000

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

        :return:
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
        :return: None
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

    while True:
        _, frame = cap.read()
        if idx % 30 == 0:
            lp_coors = cas.cascade_frame(frame)

            for x, y, w, h in lp_coors:
                match = lp_container.find_matching_lp((x, y))

                if not match:
                    lp = Lamppost(x, y, w, h, next(id_gen), color_gen(), frame)
                    lp_container.add_lp(lp)
                else:
                    match.update_location(x, y, w, h)
        else:
            for lp in lp_container.get_lps():
                if not lp.tracker_update(frame):
                    # lp_container.del_lp(lp)
                    lp.inc_decay()
                else:
                    lp_container.find_matching_lp(lp.get_coor())

        for lp in lp_container.get_lps():
            p0, p1 = lp.get_box()
            frame = cv.rectangle(frame, p0, p1, lp.get_color(), 4)
            frame = cv.putText(frame,
                               text=f"id:{lp.get_id()}",
                               org=lp.get_coor(),
                               fontFace=0,
                               fontScale=1,
                               color=lp.get_color(),
                               thickness=3)
        frame = cv.putText(frame,
                           text=f"lat: {gps_coords.lat[math.floor(idx/24)]}",
                           org=(1500, 1040),
                           fontFace=0,
                           fontScale=1,
                           color=(255, 255, 255),
                           thickness=2)
        frame = cv.putText(frame,
                           text=f"lon: {gps_coords.lon[math.floor(idx / 24)]}",
                           org=(1500, 1070),
                           fontFace=0,
                           fontScale=1,
                           color=(255, 255, 255),
                           thickness=2)

        cv.imshow("tracking", frame)
        k = cv.waitKey(1)

        if k == ord("q"):
            cv.destroyWindow("tracking")
            sys.exit(0)

        lp_container.apply_decay()
        # print([(lp.get_id(), lp.get_decay()) for lp in lp_container.get_lps()])
        idx += 1



def main():
    lp_container = Lp_container()
    cas = Cascader("models/cascade.xml")
    id_gen = number_gen()

    # cas.frame_extractor("input/Amsterdam/AMSTERDAM_OSV.mp4", "output/Amsterdam/frames/")

    files = sorted(os.listdir("output/Amsterdam/frames/"), key=lambda x: int(os.path.splitext(x)[0]))

    for idx, fn in enumerate(files):
        frame = cv.imread("output/Amsterdam/frames/" + fn)

        lp_coors = cas.cascade_frame(frame)

        for x, y, w, h in lp_coors:
            match = lp_container.find_matching_lp((x, y))

            if not match:
                lp = Lamppost(x, y, w, h, next(id_gen), color_gen())
                lp_container.add_lp(lp)
            else:
                match.update_location(x, y, w, h)

        for lp in lp_container.get_lps():
            p0, p1 = lp.get_box()
            frame = cv.rectangle(frame, p0, p1, lp.get_color(), 4)
            frame = cv.putText(frame,
                               text=f"id:{lp.get_id()}",
                               org=lp.get_coor(),
                               fontFace=2,
                               fontScale=1,
                               color=lp.get_color(),
                               thickness=3)

        # cv.imwrite(f"output/Amsterdam/detection/{idx}.jpg", frame)
        cv.imshow("tracking", frame)
        cv.waitKey(50)
        print("frame")
        # print("current lps:", [lp.get_id() for lp in lp_container.get_lps()])
        lp_container.apply_decay()


if __name__ == "__main__":
    demo()
