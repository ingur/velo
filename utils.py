from typing import Iterator
import cv2 as cv
import numpy as np

X0 = 155000
Y0 = 463000
PHI0 = 52.15517440
LAM0 = 5.38720621


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


def detect_gray_frame(frame):
    img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    img = cv.medianBlur(img, 5)

    perc = 0.004
    hist = cv.calcHist([img], [0], None, [256], [0, 256]).flatten()

    total = img.shape[0] * img.shape[1]
    target = perc * total

    summed = 0
    thresh = 0
    for i in range(255, 0, -1):
        summed += int(hist[i])
        if summed >= target:
            thresh = i
            break

    ret = cv.threshold(img, thresh < 255 and thresh or 254, 0, cv.THRESH_TOZERO)[1]
    ret = ret[0:300]
    cv.imshow("poly", ret)
    cv.waitKey(1)

    contours = cv.findContours(ret, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    # contours = sorted(contours, key=cv.contourArea, reverse=True)

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)

    detected = []
    ret = cv.cvtColor(ret, cv.COLOR_GRAY2RGB)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 0.03 * cv.arcLength(c, True), True)
        area = cv.contourArea(c)
        if len(contours_poly) > 10 and area > 1000:
            ret = cv.polylines(ret, [contours_poly[i]], True, (0, 0, 255))
            detected.append(cv.boundingRect(contours_poly[i]))
        # centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])
        # print(approx)
        # area = cv.contourArea(c)
    cv.imshow("poly", ret)
    cv.waitKey(1)

    return detected


def detect_frame(im):
    img = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    img = cv.medianBlur(img, 5)
    kernel = np.ones((12, 12), np.uint8)
    img = cv.erode(img, kernel, iterations=3)
    img = cv.dilate(img, kernel, iterations=3)

    perc = 0.015
    hist = cv.calcHist([img[:324]], [0], None, [256], [0, 256]).flatten()

    total = img.shape[0] * img.shape[1]
    target = perc * total

    summed = 0
    thresh = 0
    for i in range(255, 0, -1):
        summed += int(hist[i])
        if summed >= target:
            thresh = i
            break

    ret = cv.threshold(img, thresh < 255 and thresh or 254, 0, cv.THRESH_TOZERO)[1]
    cv.imshow("poly", ret)
    cv.waitKey(1)

    contours = cv.findContours(ret[:, :960], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = sorted(contours, key=cv.contourArea, reverse=True)

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)

    detected = []

    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 0.03 * cv.arcLength(c, True), True)
        if len(contours_poly[i]) > 2:
            detected.append(cv.boundingRect(contours_poly[i]))

        # im = cv.polylines(im, [contours_poly[i]], True, (0, 255, 255))
        # if len(approx) > 5:
        #     detected.append(boundRect[i])
        # centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])
        # approx = cv.approxPolyDP(c, 0.0001*cv.arcLength(c, True), True)
        # print(approx)
        # area = cv.contourArea(c)

    return detected

