import argparse
import datetime
import imutils
import time
import cv2

import numpy as np

vLeft = cv2.VideoCapture("./left.mpeg")
vRight = cv2.VideoCapture("./right.mpeg")

sbm = cv2.StereoBM_create()
sbm.setBlockSize(9)
sbm.setNumDisparities(112)
sbm.setPreFilterSize(5)
sbm.setPreFilterCap(1)
sbm.setMinDisparity(0)
sbm.setTextureThreshold(5)
sbm.setUniquenessRatio(5)
sbm.setSpeckleWindowSize(0)
sbm.setSpeckleRange(20)
sbm.setDisp12MaxDiff(64)

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

    disp = np.empty(frameLeft.shape)
    disp8 = np.empty(frameLeft.shape)
    
    sbm.compute(frameLeft, frameRight, disp)
    cv2.normalize(disp, disp8, 0.1, 255, cv2.NORM_MINMAX, cv2.CV_8U)

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
