import math
import sys
from types import FunctionType

import cv2 as cv
import gpxpy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial.distance import cdist

from lp_cascade import Cascader
from utils import number_gen, color_gen, wgs_to_rd, rd_to_wgs, detect_gray_frame

DEG2RAD = math.pi / 180
RAD2DEG = 180 / math.pi

# default parameters for the CSRT tracker from opencv
default_params = {
    'padding': 3.,
    'template_size': 200.,
    'gsl_sigma': 1.,
    'hog_orientations': 9.,
    'num_hog_channels_used': 18,
    'hog_clip': 2.0000000298023224e-01,
    'use_hog': 1,
    'use_color_names': 1,
    'use_gray': 1,
    'use_rgb': 0,
    'window_function': 'hann',
    'kaiser_alpha': 3.7500000000000000e+00,
    'cheb_attenuation': 45.,
    'filter_lr': 1.9999999552965164e-02,
    'admm_iterations': 4,
    'number_of_scales': 100,
    'scale_sigma_factor': 0.25,
    'scale_model_max_area': 512.,
    'scale_lr': 2.5000000372529030e-02,
    'scale_step': 1.02,
    'use_channel_weights': 1,
    'weights_lr': 1.9999999552965164e-02,
    'use_segmentation': 1,
    'histogram_bins': 16,
    'background_ratio': 2,
    'histogram_lr': 3.9999999105930328e-02,
    'psr_threshold': 3.5000000149011612e-02,
}

param_handler = cv.TrackerCSRT_Params()
# threshold for determining when the target is lost
params = {'psr_threshold': 1e-1}
for key, val in params.items():
    setattr(param_handler, key, val)


class Lamppost:
    """
    Class for tracking lampposts.
    """

    def __init__(self, x: int, y: int, w: int, h: int, lp_id: int, color: tuple, frame: np.ndarray, gps: tuple,
                 bearing):
        """
        Constructor method.

        :param x: x coordinate of bounding box
        :param y: y coordinate of bounding box
        :param w: width of bounding box
        :param h: height of bounding box
        :param lp_id: id of lamppost
        :param color: color of its bounding box in (r,g,b) format
        :param frame: frame in which the lamppost was found
        :param bearing: initial bearing of the camera in degrees
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
        # final location predicted
        self.location = None

        self.tracker = cv.TrackerCSRT_create(param_handler)
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
        Return location for printing distance on an image frame

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

    def get_angle(self, bearing) -> tuple:
        """
        Returns the angle between camera and lamppost in radians according to
        the x and y coordinates  of the bounding box.

        formula: result = ((x - minX) / (maxX - minX)) * (maxFOV - minFOV) + minFOV

        :return: angle between camera in lamppost in radians in (x, y) format
        :rtype: tuple
        """
        # deg_alpha = bearing - ((self.x + (self.w * 0.5)) / 1920) * 90.9 - 45.45
        # deg_beta = (((1080 - self.y) + (self.h * 0.5)) / 1080) * 53.6 - 26.8
        deg_alpha = (960 - (self.x + (self.w * 0.5)) / 1920) * 90.9
        deg_beta = (540 - self.y + (self.h * 0.5)) / 540 * (26.8 * 1.5)

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
        # print(self.get_id())

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

        ls = np.linalg.lstsq(R, q, rcond=None)
        p = ls[0]

        self.location = p
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



class Lp_container:
    """
    Container for the currently tracked Lamppost objects in a frame.
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

    def add_detected_lp(self, lp: Lamppost):
        """
        Adds a lamppost to the set of detected lampposts in a frame.
        Should be used if the bounding box of a lamppost is updated
        through the tracker but not the detector.

        :param lp: lamppost to add to the set
        """
        self.det_lps.add(lp)

    def find_matching_lp(self, det_lp: tuple) -> Lamppost:
        """
        Finds the closest matching known lamppost to the newly
        detected lamppost according to their location on the
        image plane or None if there is no match.

        :param det_lp: (x, y) coordinate of detected lamppost on the image plane
        to check against know lampposts
        :return: closest matching lamppost object or None if none are found
        :rtype: Lamppost
        """

        # the max distance in pixels for it to be a valid match
        max_dist = 200

        if len(self.lps) == 0:
            return None

        # generate a list with the coordinates of all know lampposts on the image plane
        coors = np.asarray([lp.get_coor() for lp in self.lps])

        dists = np.sum(np.sqrt((coors - np.asarray(det_lp)) ** 2), axis=1)

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


class Cardinal:
    """
    Umbrella class for Lamppost and Lp_container classes
    """

    def __init__(self, video_fn: str, gpx_fn: str, detector: FunctionType, data_df: pd.DataFrame = None):
        """
        constructor method

        :param video_fn: location of video file to analyse
        :param gpx_fn: location of gpx file
        :param detector: function to detect lampposts. it should be a function that takes a single frame as input
        and returns a list of bounding boxes in the form of [[x, y, width, height]]
        :param data_df: pandas dataframe containing three columns of: (lamppost_id, x-coordinate, y_coordinate)
        """

        self.video = cv.VideoCapture(video_fn)
        self.detector = detector
        self.id_gen = number_gen()
        self.data_df = data_df

        with open(gpx_fn, "r") as file:
            gpx = gpxpy.parse(file)
        segments = gpx.tracks[0].segments[0]
        self.gps_coords = pd.DataFrame([
            {'lat': p.latitude,
             'lon': p.longitude,
             'ele': p.elevation,
             'time': p.time} for p in segments.points])

        self.container = Lp_container()
        self.lampposts = []

    def analyze_frame(self, frame: np.ndarray, detect: bool, rdx: int, rdz: int, bearing: float):
        """
        Takes a single frame of video to detect lampposts in.

        :param frame: frame of video as returned by opencv
        :param detect: whether or not to query the detector
        :param rdx: x rijksdriehoekscoordinaat
        :param rdz: z rijksdriehoekcoordinaat
        :param bearing: bearing of camera in degrees
        """

        # estimated height of the camera on the bike in meters, should be checked
        rdy = 1.2

        if detect:
            lp_coors = self.detector(frame)

            for x, y, w, h in lp_coors:
                match = self.container.find_matching_lp((x, y))

                if not match:
                    lp = Lamppost(x, y, w, h, next(self.id_gen), color_gen(), frame, (rdx, rdz, rdy), bearing)
                    self.container.add_lp(lp)
                    self.add_lp(lp)

                else:
                    match.add_loc((rdx, rdz, rdy))
                    match.get_angle(bearing)

        else:
            for lp in self.container.get_lps():
                if not lp.tracker_update(frame):
                    lp.inc_decay()

                else:
                    self.container.add_detected_lp(lp)
                    lp.add_loc((rdx, rdz, rdy))
                    lp.get_angle(bearing)

    def analysis(self, video_start: int = 0, detect_rate: int = 12, debug: bool = True):
        """
        Method to handle the analysis on the video file by detecting and tracking
        lampposts throughout each frame. These detected lampposts are stored in
        self.lampposts

        :param video_start: at which second to start the video analysis
        :param detect_rate: rate at which to query detector
        :param debug: whether or not to show the results of the: detector, tracker
        and distance estimation in an image window
        """
        print("Starting video analysis...")
        # estimated height of the camera on the bike in meters, should be checked
        rdy = 1.2
        # frames per second of our camera
        fps = 24

        lats = [self.gps_coords.lat[video_start]]
        lons = [self.gps_coords.lon[video_start]]

        frame_idx = video_start * fps
        total_frames = int(self.video.get(cv.CAP_PROP_FRAME_COUNT))
        self.video.set(cv.CAP_PROP_POS_FRAMES, frame_idx)

        while True:
            ret, frame = self.video.read()

            if not ret:
                print(f"Could not read video file beyond frame {frame_idx}")
                break

            # check to see if we started the video from a higher timestamp than 0
            idx = 0 if video_start > 0 else frame_idx % detect_rate
            rdx, rdz = wgs_to_rd(lats[idx], lons[idx])

            if idx == 0:

                # interpolate the coordinates between two gpx data-points
                lats = np.linspace(self.gps_coords.lat[math.floor(frame_idx / fps)],
                                   self.gps_coords.lat[math.floor(frame_idx / fps + 1)], num=detect_rate)
                lons = np.linspace(self.gps_coords.lon[math.floor(frame_idx / fps)],
                                   self.gps_coords.lon[math.floor(frame_idx / fps + 1)], num=detect_rate)

                rdx, rdz = wgs_to_rd(lats[idx], lons[idx])
                # calculate bearing in degrees
                p0 = wgs_to_rd(lats[0], lons[0])
                p1 = wgs_to_rd(lats[-1], lons[-1])
                bearing = math.atan2(p1[1] - p0[1], p1[0] - p0[0]) * RAD2DEG

                self.analyze_frame(frame, True, rdx, rdz, bearing)
            else:
                self.analyze_frame(frame, False, rdx, rdz, bearing)

            if debug:
                for lp in self.container.get_lps():
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

                    lp_rdy = np.clip(lp_rdy, 2, 8)
                    dist = round(np.sqrt((lp_rdx - rdx) ** 2 + (lp_rdz - rdz) ** 2 + (lp_rdy - rdy) ** 2)[0] / 10, 2)
                    frame = cv.putText(frame,
                                       text=f"dist: {dist}",
                                       org=lp.get_dist_loc(),
                                       fontFace=0,
                                       fontScale=1,
                                       color=lp.get_color(),
                                       thickness=3)

                frame = cv.putText(frame,
                                   text=f"lat: {lats[idx % detect_rate]}",
                                   org=(1500, 1040),
                                   fontFace=0,
                                   fontScale=1,
                                   color=(255, 255, 255),
                                   thickness=2)
                frame = cv.putText(frame,
                                   text=f"lon: {lons[idx % detect_rate]}",
                                   org=(1500, 1070),
                                   fontFace=0,
                                   fontScale=1,
                                   color=(255, 255, 255),
                                   thickness=2)

                cv.imshow("Debug window", frame)
                k = cv.waitKey(1)

                if k == ord("q"):
                    cv.destroyWindow("Debug window")
                    sys.exit(0)

            video_start = 0
            frame_idx += 1
            self.container.apply_decay()
            print(f"Analysed {frame_idx} / {total_frames} frames", end='\r')
        print("Video analysed")

    def add_lp(self, lp: Lamppost):
        """
        Adds a lamppost to self.lampposts

        :param lp: lamppost object to store
        """
        self.lampposts.append(lp)

    def get_lps(self) -> list:
        """
        Returns the list of lamppost

        :return: list with all detected lampposts in the video
        :rtype list:
        """
        return self.lampposts

    def get_relevant_lps(self):
        """
        Reduces the size of the pandas dataframe containing all the lampposts and
        restricts it to the lampposts who are situated along the route of the
        cyclist.
        """
        print("fetching data...")
        rds_route = np.array([wgs_to_rd(gps['lat'], gps['lon']) for _, gps in self.gps_coords[['lat', 'lon']].iterrows()], dtype=np.int64)
        rds_data = self.data_df[['X', 'Y']].to_numpy()
        # remove all rows containing a nan
        rds_data = rds_data[~np.isnan(rds_data).any(axis=1), :]
        print("computing distances...")
        dist_matrix = cdist(rds_data, rds_route, 'euclidean')
        # get unique indices of which coordinates follow the condition
        indices = np.unique(np.argwhere(dist_matrix < 10)[:, 0])
        # remove indices of lampposts that are too far away
        self.data_df = self.data_df.iloc[indices]

    def delete_redundant_lps(self):
        """
        Lampposts who are too close to each other are combined
        by averaging their coordinates.
        """

        # distance in meters for when two lampposts should be considered as the same
        min_dist = 2

        # get the first 2 columns of the lampposts coordinates (rdx, rdz)
        coors = np.vstack([lp.intersect().reshape(3) for lp in self.lampposts])[:, :2]
        # compute the inter-lamppost distance
        dist_mat = cdist(coors, coors, 'euclidean')
        indices = np.argwhere(dist_mat < min_dist)

        # key is the index of the lps and its values are the indices of lps close to it
        neighboring_lps = dict()
        # set of all lps that will be merged into other lps
        redundant_lps = set()

        for lp1, lp2 in indices:
            if lp1 in redundant_lps:
                continue

            if lp1 in neighboring_lps:
                neighboring_lps[lp1].append(lp2)
            else:
                neighboring_lps[lp1] = [lp2]

                redundant_lps.add(lp2)

        print(neighboring_lps)
        print(len(self.lampposts))
        # recalculate locations
        for lp, red_lps in neighboring_lps.items():
            mean_loc = np.mean(coors[red_lps])
            self.lampposts[lp].location = mean_loc

        # remove redundant lampposts from memory
        for idx in sorted(redundant_lps, reverse=True):
            del self.lampposts[idx]

    def plot_lampposts(self):
        """
        This plots the detected lampposts and know lampposts
        (if provided during initialization) on a Mapbox map.
        """

        ids = []
        lats_route = []
        lons_route = []

        lats_data = []
        lons_data = []

        pk = "pk.eyJ1IjoibWFlZ29yaSIsImEiOiJja3NybzN2eWowaGJ2MnZwbmp3MTd5NDhlIn0.jeweG30DDP_Fzj-KRJ7OiQ"

        # get lamppost locations from provided by Amsterdam
        if isinstance(self.data_df, pd.DataFrame):
            for idx in range(len(self.data_df)):
                lp_id, x, y = self.data_df.iloc[idx]
                lat, lon = rd_to_wgs(x, y)

                ids.append(lp_id)
                lats_data.append(lat)
                lons_data.append(lon)

        # get lamppost locations from detections
        for lp in self.get_lps():
            rdx, rdz, _ = lp.intersect()
            lat, lon = rd_to_wgs(rdx, rdz)

            lats_route.append(lat[0])
            lons_route.append(lon[0])

        fig = go.Figure(go.Scattermapbox(
            lat=lats_data,
            lon=lons_data,
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=9
            ),
            hoverinfo=['lat', 'lon', 'text'],
            text=ids
        ))

        fig.add_trace(go.Scattermapbox(
            lat=lats_route,
            lon=lons_route,
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=9,
                color='red'
            )
        ))

        fig.update_layout(autosize=True,
                          mapbox_style="open-street-map",
                          hovermode='closest',
                          mapbox=dict(
                              accesstoken=pk,
                              bearing=0,
                              center=go.layout.mapbox.Center(
                                  lat=52.377956,
                                  lon=4.897070
                              ),
                              pitch=0,
                              zoom=10
                          )
                          )

        fig.show()


def demo():
    cas = Cascader("models/cascade.xml")
    video_fn = "input/Amsterdam/AMSTERDAM_OSV.mp4"
    gpx_fn = "input/Amsterdam/AMSTERDAM_OSV.gpx"
    exc_fn = "input/Amsterdam/Bijlage 5 Assetoverzicht OVL te inspecteren lichtpunten.xlsx"
    detector = cas.cascade_frame

    data_df = pd.read_excel(exc_fn)[["Identificatie", "X", "Y"]]

    cardinal = Cardinal(video_fn, gpx_fn, detector, data_df)
    cardinal.analysis(video_start=0, debug=True)
    cardinal.get_relevant_lps()
    cardinal.delete_redundant_lps()
    cardinal.plot_lampposts()
    # cardinal.analysis(video_start=0, debug=False)
    #
    # for lp in cardinal.get_lps():
    #     print(lp.get_id())


if __name__ == "__main__":
    demo()
