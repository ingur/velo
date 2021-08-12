import sys
import numpy as np
import cv2 as cv
import os
import argparse

parser = argparse.ArgumentParser(description="Apply cascade filter to detect lampposts")
parser.add_argument("-vid",
                    type=str,
                    help="path to video file to apply the cascade filter to",
                    default="input/Amsterdam/AMSTERDAM_OSV.mp4",
                    required=False)
parser.add_argument("-sav",
                    type=str,
                    help="path to store the individual frames of the video",
                    default="output/Amsterdam/frames/",
                    required=False)
parser.add_argument("-cas",
                    type=str,
                    help="path to cascade xml file",
                    default="models/cascade.xml",
                    required=False)
parser.add_argument("-save_results",
                    type=bool,
                    help="whether to save the results in a numpy file",
                    default=True,
                    required=False)


def frame_extractor(vid_path: str, sav_path: str, hop=1):
    """
    Extracts a video file into its individual frames, each
    frame is named after the second in which it occurs and
    saved to img_path.

    :param vid_path: Path to video file which to extract the frames from
    :param sav_path: Path to save the individual frames to
    :param hop: the amount of frames to skip between saving
    :return: None
    """

    cap = cv.VideoCapture(vid_path)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv.CAP_PROP_FPS)
    print(f"{vid_path} has {total_frames} frames, which will be reduced to {int(total_frames / hop)}")

    # skip this function if the frames are already present
    if len(os.listdir(sav_path)) == round(total_frames / hop) + 1:
        print("Video already split into frames")
        return

    # splitting the frames
    print("Splitting video frames...")
    for i in range(total_frames):
        state, frame = cap.read()
        # if i % hop == 0:
        if not state:
            print(f"Cannot extract frame {i}, exiting")
            sys.exit()
        cv.imwrite(f"{sav_path}{round(i)}.jpg", frame)

    print("Finished splitting frames")
    return

class Cascader:
    """
    Handles the Haar cascade classifier to detect lampposts
    """

    def __init__(self, cas_path: str):
        """
        Constructor method

        :param cas_path: path to cascade xml file
        """

        self.cas = cv.CascadeClassifier(cas_path)

    def cascade_video(self, vid_path: str, sav_path: str, save_results=False) -> list:
        """
        Returns the location of detected lampposts in each frame in a
        (x, y, width, height) format.

        :param vid_path: path to video file to draw on
        :param sav_path: destination path for the individual frames
        :param save_results: whether to save the results in a numpy file
        :return: list of lamppost
        """

        # split video into individual frames
        detected = []

        # apply cascade filter to all available frames
        print("Applying cascade filter...")
        for f in os.listdir(sav_path):
            frame = cv.imread(sav_path + f)
            detected.append(self.cas.detectMultiScale(frame, minNeighbors=90))

        if save_results:
            city = sav_path.split("/")[:2]
            dest = "/".join(city)

            with open(dest + "/detected_lps.npy", 'wb') as f:
                np.save(f, np.array(detected))

            print(f"saved results to {dest}/detected_lps.npy")

        return detected

    def frame_extractor(self, vid_path, sav_path, hop=1):
        """
        Extracts a video file into its individual frames, each
        frame is named after the second in which it occurs and
        saved to img_path.

        :param vid_path: Path to video file which to extract the frames from
        :param sav_path: Path to save the individual frames to
        :param hop: the amount of frames to skip between saving
        :return: None
        """

        cap = cv.VideoCapture(vid_path)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv.CAP_PROP_FPS)
        print(f"{vid_path} has {total_frames} frames, which will be reduced to {int(total_frames / hop)}")

        # skip this function if the frames are already present
        if len(os.listdir(sav_path)) == round(total_frames / hop):
            print("Video already split into frames")
            return

        # splitting the frames
        print("Splitting video frames...")
        for i in range(total_frames):
            state, frame = cap.read()
            # if i % hop == 0:
            if not state:
                print(f"Cannot extract frame {i}, exiting")
                sys.exit()
            cv.imwrite(f"{sav_path}{round(i)}.jpg", frame)

        print("Finished splitting frames")
        return

    def cascade_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns the locations of detected lampposts in a
        single frame

        :param frame: numpy.ndarray of an image as read by opencv
        :param cas_path: path to cascade.xml file
        :return: numpy array with the locations of all detected lampposts on the image plane
        :rtype: np.ndarray
        """

        return self.cas.detectMultiScale(frame, minNeighbors=90)


if __name__ == "__main__":
    args = parser.parse_args()
    vid_path = args.vid
    sav_path = args.sav
    cas_path = args.cas
    save_results = args.save_results

    if not os.path.exists(vid_path):
        print(f"No video file found at {vid_path}, exiting...")
        sys.exit(1)

    if not os.path.exists(sav_path):
        print(f"{sav_path} not found, exiting...")
        sys.exit(1)

    if not os.path.exists(cas_path):
        print(f"cascade xml file not found at {cas_path}, exiting...")
        sys.exit(1)

    lps = cascade_video(vid_path, sav_path, cas_path, save_results)
