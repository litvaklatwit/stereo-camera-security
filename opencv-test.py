import argparse
import datetime
import imutils
import time
import cv2

import numpy as np

vLeft = cv2.VideoCapture("./left.mpeg")
vRight = cv2.VideoCapture("./right.mpeg")

def processFrame():
    frameLeft = vLeft.read()[1]
    frameRight = vRight.read()[1]

    if frameLeft is None or frameRight is None:
        return False

	# resize the frame, convert it to grayscale, and blur it
    #frameLeft = imutils.resize(frameLeft, width=500)
    #gray = cv2.cvtColor(frameLeft, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(gray, (21, 21), 0)

    frameLeft = cv2.cvtColor(frameLeft, cv2.COLOR_BGR2GRAY)
    frameRight = cv2.cvtColor(frameRight, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT()
    kp1, des1 = sift.detectAndCompute(frameLeft, None)
    kp2, des2 = sift.detectAndCompute(frameRight, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k = 2)

    good = []
    pts1 = []
    pts2 = []
    

    cv2.imshow("Left", frameLeft)
    cv2.imshow("Right", frameRight)
    cv2.imshow("Disparity", disp)

    return True


#while True:
#    if not processFrame():
#        break

#    key = cv2.waitKey(16) & 0xFF

#    if key == ord("q"):
#        break

processFrame()
cv2.waitKey(0)
