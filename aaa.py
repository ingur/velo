import os
from typing import Iterator
import cv2 as cv
import imutils
import numpy as np
from lp_cascade import Cascader
import sys
import gpxpy
import pandas as pd
import math
from imutils import contours
from skimage import measure
import random as rng

cv2 = cv

cap = cv.VideoCapture("input/Amsterdam/AMSTERDAM_OSV.mp4")
idx = 0

filterSize =(3, 3)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                   filterSize)

while True:
    _, im = cap.read()

    if idx % 1 == 0:
        img = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        img = cv.medianBlur(img, 5)
        kernel = np.ones((12,12),np.uint8)
        img = cv.erode(img, kernel, iterations=3)
        img = cv.dilate(img,kernel,iterations=3)

        # cv.imshow("AAAAAAAA", img)

        perc = 0.01
        hist = cv.calcHist([img], [0], None, [256], [0, 256]).flatten()

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
        # circles = cv.HoughCircles(ret, cv.HOUGH_GRADIENT, 1, 200, 200, 100, 50, 200)
        # print(circles)
        # if circles is not None:
        #     circles = np.uint16(np.around(circles))
        #     print(circles)
        #     for i in circles[0, :]:
        #         print(i)
        #         center = (i[0], i[1])
        #         # circle center
        #         cv.circle(im, center, 1, (0, 100, 100), 3)
        #         # circle outline
        #         radius = i[2]
        #         cv.circle(im, center, radius, (255, 0, 255), 3)
        # print(ret)
        # print(thresh)
        # print(ret)

        # tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        # ret, thresh = cv2.threshold(tophat, thresh, 255, cv2.THRESH_BINARY)
        # print(np.amax(tophat))


        # im[ret > 0] = [255, 0, 0]
        # cv.imshow("oek", im)

        contours = cv2.findContours(ret, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # for c in contours:
            # Highlight largest contour
            # cv2.drawContours(im, [c], -1, (36,255,12), 3)
        # Approximate contours to polygons + get bounding rects and circles
        contours_poly = [None]*len(contours)
        boundRect = [None]*len(contours)
        centers = [None]*len(contours)
        radius = [None]*len(contours)
        for i, c in enumerate(contours):
            contours_poly[i] = cv.approxPolyDP(c, 3, True)
            boundRect[i] = cv.boundingRect(contours_poly[i])
            centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])

        for i in range(len(contours[:1])):
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            color = (0, 255, 0)
            if centers[i][1] <= 0.3 * 1080 and radius[i] < 100:
                cv.rectangle(im, (int(boundRect[i][0]), int(boundRect[i][1])), \
                             (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
                # cv.circle(im, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
        cv.imshow("oek", im)



        # Setup SimpleBlobDetector parameters.
        # params = cv2.SimpleBlobDetector_Params()
        #
        # # Change thresholds
        # params.minThreshold = 10
        # params.maxThreshold = 200
        #
        #
        # # Filter by Area.
        # params.filterByArea = True
        # params.minArea = 1500
        #
        # # Filter by Circularity
        # params.filterByCircularity = True
        # params.minCircularity = 0.1
        #
        # # Filter by Convexity
        # params.filterByConvexity = True
        # params.minConvexity = 0.87
        #
        # # Filter by Inertia
        # params.filterByInertia = True
        # params.minInertiaRatio = 0.01
        #
        # # Create a detector with the parameters
        # # OLD: detector = cv2.SimpleBlobDetector(params)
        # detector = cv2.SimpleBlobDetector_create(params)
        #
        #
        # # Detect blobs.
        # keypoints = detector.detect(ret)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
        # the size of the circle corresponds to the size of blob

        # im_with_keypoints = cv2.drawKeypoints(ret, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #
        # im[cv.cvtColor(im_with_keypoints, cv.COLOR_BGR2GRAY) > 0] = [255, 0, 0]
        # cv.imshow("oek", im)
        # cv.imshow("whatisthisowo", im_with_keypoints)



# kernel = np.ones((20,20),np.uint8)
        # opening = cv.morphologyEx(ret,cv.MORPH_OPEN,kernel, iterations = 2)
        # # sure background area
        # sure_bg = cv.dilate(opening,kernel,iterations=3)
        # # Finding sure foreground area
        # dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
        # # cv.imshow("HELP", dist_transform)
        # ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
        # # Finding unknown region
        # sure_fg = np.uint8(sure_fg)
        # unknown = cv.subtract(sure_bg,sure_fg)
        #
        # # Marker labelling
        # ret, markers = cv.connectedComponents(sure_fg)
        # # Add one to all labels so that sure background is not 0, but 1
        # markers = markers+1
        # # Now, mark the region of unknown with zero
        # markers[unknown==255] = 0
        #
        # # markers = cv.watershed(im,markers)
        # # im[markers == -1] = [255,0,0]
        #
        # im[dist_transform>0] = [255, 0, 0]
        # # print(np.amax(dist_transform))
        # cv.imshow("XD", im)

    k = cv.waitKey(1)

    if k == ord("q"):
        cv.destroyWindow("tracking")
        sys.exit(0)

    idx += 1
