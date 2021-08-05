import sys
import cv2 as cv
import os
import argparse

parser = argparse.ArgumentParser(description="Apply cascade filter to detect lampposts")
parser.add_argument("vid_path", type=str, help="path to video file to apply the cascade filter to", default="video/AMSTERDAM_OSV.mp4")
parser.add_argument("img_path", type=str, help="path to store the individual frames of the video", default="frames/")
parser.add_argument("cas_path", type=str, help="path to cascade xml file", default="output/")
parser.add_argument("out_path", type=str, help="path to store ", default="cascade")
parser.add_argument("save_vid", type=bool, help="whether to save the frames as a video", required=False)
parser.add_argument("output_vid_name", type=str, help="path and name of output video file", required=False)

# vid_path = "video/AMSTERDAM_OSV.mp4"
# img_path = "frames/"
# out_path = "output/"
# cas_path = "cascade.xml"
#
# if not os.path.exists(img_path):
#     os.mkdir("frames")
# #
# # if not os.path.exists(out_path):
# #     os.mkdir("output")


def frame_extractor(vid_path, sav_path, hop=30):
    """
    Extracts a video file into its individual frames, each
    frame is named after the second in which it occurs
    vid_path: path to video file to extract
    sav_path: path to save the frames to as images
    hop     : the amount of frames to skip before saving a frame
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
        if i % hop == 0:
            if not state:
                print(f"Cannot extract frame {i}, exiting")
                sys.exit()
            cv.imwrite(f"frames/{round(i / fps)}.jpg", frame)

    print("Finished splitting frames")
    return


def cascader(vid_path: str, img_path: str, cas_path: str, out_path: str, save_video: str = False, out_video_name: str = "output") -> list:
    """
    Draws bounding boxes around lamppost from a video file
    and outputs them as individual frames.
    :param vid_path: path to video file to draw on
    :param img_path: destination path for the individual frames
    :param cas_path: path to cascade xml file
    :param out_path: destination path for the drawn frames
    :param save_video: whether to save the frames as a video file
    :param out_video_name: name of the output video
    :return: list of lamppost
    """

    # split video into individual frames
    frame_extractor(vid_path, img_path)

    # load cascade
    lp_cas = cv.CascadeClassifier(cas_path)

    # apply cascade filter to all available frames
    print("Applying cascade filter...")
    for f in os.listdir(img_path):
        frame = cv.imread(img_path + f)
        lps = lp_cas.detectMultiScale(frame)

        # continue if no lamp posts are detected in the frame
        if len(lps) == 0:
            continue

        # draw the bounding box around the light of the light post
        for (x, y, w, h) in lps:
            frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)

        cv.imwrite(out_path + f, frame)

    if save_video:
        print("Saving frames as video file...")
        output_frames = os.listdir(out_path)
        frame_array = []

        for i in range(len(output_frames)):
            frame = cv.imread(out_path + output_frames[i])
            frame_array.append(frame)

        h, w, _ = frame_array[0].shape

        out = cv.VideoWriter(f"{out_video_name}.avi", cv.VideoWriter_fourcc(*'DIVX'), 23.0, (w, h))

        for i in range(len(frame_array)):
            out.write(frame_array[i])

        out.release()


if __name__ == "__main__":
    cascader(vid_path, img_path, cas_path, out_path)
